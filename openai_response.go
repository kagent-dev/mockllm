package mockllm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/openai/openai-go/v3/responses"
)

// OpenAIResponseProvider handles OpenAI Responses API request/response mocking
type OpenAIResponseProvider struct {
	mocks []OpenAIResponseMock
}

// NewOpenAIResponseProvider creates a new OpenAI Responses API provider with the given mocks
func NewOpenAIResponseProvider(mocks []OpenAIResponseMock) *OpenAIResponseProvider {
	return &OpenAIResponseProvider{
		mocks: mocks,
	}
}

// Handle processes an OpenAI Responses API request
func (p *OpenAIResponseProvider) Handle(w http.ResponseWriter, r *http.Request) {
	// Parse request body to check for stream
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to read body: %v", err), http.StatusBadRequest)
		return
	}
	r.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))

	// Check for stream: true in raw JSON
	// ResponseNewParams type also does not contain streaming field like ChatCompletionNewParams
	var rawMap map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &rawMap); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}

	// Patch input items to add "type" field if missing
	// This works around upstream SDK bug: https://github.com/openai/openai-go/issues/465
	patchResponseInputType(rawMap)

	// Re-marshal the patched JSON
	patchedBytes, err := json.Marshal(rawMap)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to marshal patched request: %v", err), http.StatusInternalServerError)
		return
	}

	// Parse the patched request into SDK type
	var requestBody responses.ResponseNewParams
	if err := json.Unmarshal(patchedBytes, &requestBody); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}

	// Find a matching mock
	mock := p.findMatchingResponseMock(requestBody, r.Header)
	if mock == nil {
		http.Error(w, fmt.Sprintf("No matching mock found. Request: %s",
			string(bodyBytes)), http.StatusNotFound)
		return
	}

	// Return the response
	isStream := false
	if streamVal, ok := rawMap["stream"].(bool); ok {
		isStream = streamVal
	}

	if isStream {
		p.handleResponsesStreamingResponse(w, mock.Response)
	} else {
		p.handleNonStreamingResponse(w, mock.Response)
	}
}

// patchResponseInputType adds "type": "message" to input items that are missing it
// This is necessary since without this type field, the SDK will not be able to unmarshal the input into the ResponseNewParams type
func patchResponseInputType(rawMap map[string]interface{}) {
	input, ok := rawMap["input"]
	if !ok {
		return
	}

	inputList, ok := input.([]interface{})
	if !ok {
		return
	}

	for _, item := range inputList {
		itemMap, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		// Add type field if missing and item looks like a message (has role and content)
		if _, hasType := itemMap["type"]; !hasType {
			if _, hasRole := itemMap["role"]; hasRole {
				if _, hasContent := itemMap["content"]; hasContent {
					itemMap["type"] = "message"
				}
			}
		}
	}
}

// findMatchingResponseMock finds the first mock that matches the Responses API request
func (p *OpenAIResponseProvider) findMatchingResponseMock(request responses.ResponseNewParams, headers http.Header) *OpenAIResponseMock {
	for _, mock := range p.mocks {
		if p.responseRequestsMatch(mock.Match, request) && headersMatch(mock.Match.Headers, headers) {
			return &mock
		}
	}
	return nil
}

// responseRequestsMatch checks if two Responses API requests are equivalent
func (p *OpenAIResponseProvider) responseRequestsMatch(expected OpenAIResponseRequestMatch, actual responses.ResponseNewParams) bool {
	switch expected.MatchType {
	case MatchTypeExact:
		// Check if Input fields match exactly using JSON comparison
		jsonExpected, err := json.Marshal(expected.Input)
		if err != nil {
			return false
		}
		jsonActual, err := json.Marshal(actual.Input)
		if err != nil {
			return false
		}
		return bytes.Equal(jsonExpected, jsonActual)
	case MatchTypeContains:
		// Input is a union type, either OfString or OfInputItemList will be set
		// Check if expected is OfString
		if expected.Input.OfString.Valid() {
			// Expected is a string, check if actual contains it
			if actual.Input.OfString.Valid() {
				// Both are strings, do simple contains check
				return strings.Contains(actual.Input.OfString.Value, expected.Input.OfString.Value)
			}

			// Some agent frameworks like OpenAI Agents SDK will format the user input as a list of messages to be compatible with Chat Completion API
			// Therefore we also need to match a string against the last message in the input (similar to chat completion matching)
			if len(actual.Input.OfInputItemList) > 0 {
				lastItem := actual.Input.OfInputItemList[len(actual.Input.OfInputItemList)-1]
				actualBytes, err := json.Marshal(lastItem)
				if err == nil {
					return strings.Contains(string(actualBytes), expected.Input.OfString.Value)
				}
			}

			return false
		}

		// Check if expected is OfInputItemList and non empty
		expectedBytes, err := json.Marshal(expected.Input.OfInputItemList)
		if err == nil && len(expectedBytes) > 0 && string(expectedBytes) != "null" && string(expectedBytes) != "{}" {
			actualBytes, err := json.Marshal(actual.Input.OfInputItemList)
			if err != nil || len(actualBytes) == 0 || string(actualBytes) == "null" || string(actualBytes) == "{}" {
				return false
			}
			// For contains matching on item lists, check if the actual JSON contains the expected JSON
			// Ideally we would use a deep equal comparison, but ResponseInputParam is another union type so it will be complex
			return strings.Contains(string(actualBytes), string(expectedBytes))
		}

		// Neither field is set in expected, cannot match
		return false
	default:
		return false
	}
}

// handleNonStreamingResponse sends a JSON response
func (p *OpenAIResponseProvider) handleNonStreamingResponse(w http.ResponseWriter, response any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	if err := json.NewEncoder(w).Encode(response); err != nil {
		http.Error(w, fmt.Sprintf("Failed to encode response: %v", err), http.StatusInternalServerError)
	}
}

// handleResponsesStreamingResponse sends a streaming SSE response for Responses API
func (p *OpenAIResponseProvider) handleResponsesStreamingResponse(w http.ResponseWriter, response responses.Response) {
	events, err := buildResponseStream(response)
	if err != nil {
		http.Error(w, fmt.Sprintf("Invalid streaming response fixture: %v", err), http.StatusInternalServerError)
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported by server", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	for _, event := range events {
		fmt.Fprintf(w, "event: %s\n", event.eventType)
		fmt.Fprintf(w, "data: %s\n\n", event.data)
		flusher.Flush()
	}

	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// streamEvent separates semantic event construction from SSE framing.
type streamEvent struct {
	eventType string
	data      json.RawMessage
}

type outputItemStreamPayload struct {
	Type           string `json:"type"`
	SequenceNumber int64  `json:"sequence_number"`
	OutputIndex    int64  `json:"output_index"`
	Item           any    `json:"item"`
}

type contentPartStreamPayload struct {
	Type           string `json:"type"`
	SequenceNumber int64  `json:"sequence_number"`
	ItemID         string `json:"item_id"`
	OutputIndex    int64  `json:"output_index"`
	ContentIndex   int64  `json:"content_index"`
	Part           any    `json:"part"`
}

type messageStreamItem struct {
	ID      string `json:"id"`
	Type    string `json:"type"`
	Status  string `json:"status"`
	Role    string `json:"role"`
	Content any    `json:"content"`
}

type functionCallStreamItem struct {
	ID        string `json:"id"`
	Type      string `json:"type"`
	Status    string `json:"status"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Namespace string `json:"namespace,omitempty"`
	Arguments string `json:"arguments"`
}

type outputTextStreamPart struct {
	Type        string                                        `json:"type"`
	Text        string                                        `json:"text"`
	Annotations []responses.ResponseOutputTextAnnotationUnion `json:"annotations"`
	Logprobs    []responses.ResponseOutputTextLogprob         `json:"logprobs,omitempty"`
}

// buildResponseStream translates one final Responses API fixture into the
// lifecycle events emitted by the real streaming API. It never changes response.
func buildResponseStream(response responses.Response) ([]streamEvent, error) {
	if response.ID == "" {
		return nil, fmt.Errorf("response id is required")
	}

	inProgress := response
	inProgress.Status = responses.ResponseStatusInProgress
	inProgress.Output = []responses.ResponseOutputItemUnion{}

	completed := response
	completed.Status = responses.ResponseStatusCompleted

	sequence := int64(0)
	events := make([]streamEvent, 0)
	add := func(eventType string, payload any) error {
		data, err := json.Marshal(payload)
		if err != nil {
			return fmt.Errorf("marshal %s: %w", eventType, err)
		}
		events = append(events, streamEvent{eventType: eventType, data: data})
		sequence++
		return nil
	}

	if err := add("response.created", responses.ResponseCreatedEvent{SequenceNumber: sequence, Response: inProgress}); err != nil {
		return nil, err
	}
	if err := add("response.in_progress", responses.ResponseInProgressEvent{SequenceNumber: sequence, Response: inProgress}); err != nil {
		return nil, err
	}

	for outputIndex, outputItem := range response.Output {
		switch outputItem.Type {
		case "message":
			message := outputItem.AsMessage()
			if message.ID == "" {
				return nil, fmt.Errorf("output[%d] message id is required", outputIndex)
			}
			for contentIndex, contentItem := range message.Content {
				if contentItem.Type != "output_text" {
					return nil, fmt.Errorf("output[%d].content[%d] type %q is unsupported", outputIndex, contentIndex, contentItem.Type)
				}
			}

			addedItem := messageStreamItem{ID: message.ID, Type: "message", Status: "in_progress", Role: "assistant", Content: []any{}}
			if err := add("response.output_item.added", outputItemStreamPayload{Type: "response.output_item.added", SequenceNumber: sequence, OutputIndex: int64(outputIndex), Item: addedItem}); err != nil {
				return nil, err
			}

			for contentIndex, contentItem := range message.Content {
				content := contentItem.AsOutputText()
				addedPart := outputTextStreamPart{Type: "output_text", Text: "", Annotations: []responses.ResponseOutputTextAnnotationUnion{}}
				if err := add("response.content_part.added", contentPartStreamPayload{Type: "response.content_part.added", SequenceNumber: sequence, ItemID: message.ID, OutputIndex: int64(outputIndex), ContentIndex: int64(contentIndex), Part: addedPart}); err != nil {
					return nil, err
				}
				if err := add("response.output_text.delta", responses.ResponseTextDeltaEvent{SequenceNumber: sequence, ItemID: message.ID, OutputIndex: int64(outputIndex), ContentIndex: int64(contentIndex), Delta: content.Text, Logprobs: []responses.ResponseTextDeltaEventLogprob{}}); err != nil {
					return nil, err
				}
				if err := add("response.output_text.done", responses.ResponseTextDoneEvent{SequenceNumber: sequence, ItemID: message.ID, OutputIndex: int64(outputIndex), ContentIndex: int64(contentIndex), Text: content.Text, Logprobs: []responses.ResponseTextDoneEventLogprob{}}); err != nil {
					return nil, err
				}
				donePart := outputTextStreamPart{Type: "output_text", Text: content.Text, Annotations: content.Annotations, Logprobs: content.Logprobs}
				if err := add("response.content_part.done", contentPartStreamPayload{Type: "response.content_part.done", SequenceNumber: sequence, ItemID: message.ID, OutputIndex: int64(outputIndex), ContentIndex: int64(contentIndex), Part: donePart}); err != nil {
					return nil, err
				}
			}

			doneItem := messageStreamItem{ID: message.ID, Type: "message", Status: "completed", Role: "assistant", Content: message.Content}
			if err := add("response.output_item.done", outputItemStreamPayload{Type: "response.output_item.done", SequenceNumber: sequence, OutputIndex: int64(outputIndex), Item: doneItem}); err != nil {
				return nil, err
			}

		case "function_call":
			functionCall := outputItem.AsFunctionCall()
			var rawFunctionCall struct {
				Namespace string `json:"namespace"`
			}
			if err := json.Unmarshal([]byte(outputItem.RawJSON()), &rawFunctionCall); err != nil {
				return nil, fmt.Errorf("decode output[%d] function call: %w", outputIndex, err)
			}
			if functionCall.ID == "" {
				return nil, fmt.Errorf("output[%d] function call id is required", outputIndex)
			}
			if functionCall.CallID == "" || functionCall.Name == "" {
				return nil, fmt.Errorf("output[%d] function call requires call_id and name", outputIndex)
			}
			if !json.Valid([]byte(functionCall.Arguments)) {
				return nil, fmt.Errorf("output[%d] function call arguments must be valid JSON", outputIndex)
			}

			addedItem := functionCallStreamItem{ID: functionCall.ID, Type: "function_call", Status: "in_progress", CallID: functionCall.CallID, Name: functionCall.Name, Namespace: rawFunctionCall.Namespace, Arguments: ""}
			if err := add("response.output_item.added", outputItemStreamPayload{Type: "response.output_item.added", SequenceNumber: sequence, OutputIndex: int64(outputIndex), Item: addedItem}); err != nil {
				return nil, err
			}
			if err := add("response.function_call_arguments.delta", responses.ResponseFunctionCallArgumentsDeltaEvent{SequenceNumber: sequence, ItemID: functionCall.ID, OutputIndex: int64(outputIndex), Delta: functionCall.Arguments}); err != nil {
				return nil, err
			}
			if err := add("response.function_call_arguments.done", responses.ResponseFunctionCallArgumentsDoneEvent{SequenceNumber: sequence, ItemID: functionCall.ID, OutputIndex: int64(outputIndex), Arguments: functionCall.Arguments, Name: functionCall.Name}); err != nil {
				return nil, err
			}
			doneItem := functionCallStreamItem{ID: functionCall.ID, Type: "function_call", Status: "completed", CallID: functionCall.CallID, Name: functionCall.Name, Namespace: rawFunctionCall.Namespace, Arguments: functionCall.Arguments}
			if err := add("response.output_item.done", outputItemStreamPayload{Type: "response.output_item.done", SequenceNumber: sequence, OutputIndex: int64(outputIndex), Item: doneItem}); err != nil {
				return nil, err
			}

		case "tool_search_call":
			rawToolSearchCall := json.RawMessage(outputItem.RawJSON())
			var toolSearchCall struct {
				ID     string `json:"id"`
				CallID string `json:"call_id"`
			}
			if err := json.Unmarshal(rawToolSearchCall, &toolSearchCall); err != nil {
				return nil, fmt.Errorf("decode output[%d] tool search call: %w", outputIndex, err)
			}
			if toolSearchCall.ID == "" || toolSearchCall.CallID == "" {
				return nil, fmt.Errorf("output[%d] tool search call requires id and call_id", outputIndex)
			}
			if err := add("response.output_item.done", outputItemStreamPayload{Type: "response.output_item.done", SequenceNumber: sequence, OutputIndex: int64(outputIndex), Item: rawToolSearchCall}); err != nil {
				return nil, err
			}

		default:
			return nil, fmt.Errorf("output[%d] type %q is unsupported for streaming", outputIndex, outputItem.Type)
		}
	}

	if err := add("response.completed", responses.ResponseCompletedEvent{SequenceNumber: sequence, Response: completed}); err != nil {
		return nil, err
	}
	return events, nil
}
