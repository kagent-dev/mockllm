package mockllm

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func responseFixture(t *testing.T, outputJSON string) responses.Response {
	t.Helper()
	data := `{
		"id":"resp_test","object":"response","created_at":1,"model":"gpt-5.2-codex",
		"status":"completed","output":` + outputJSON + `,
		"parallel_tool_calls":false,"tools":[]
	}`
	var response responses.Response
	require.NoError(t, json.Unmarshal([]byte(data), &response))
	return response
}

func decodeStreamEvents(t *testing.T, events []streamEvent) []map[string]any {
	t.Helper()
	decoded := make([]map[string]any, len(events))
	for i, event := range events {
		require.NoError(t, json.Unmarshal(event.data, &decoded[i]))
		assert.Equal(t, event.eventType, decoded[i]["type"])
		assert.Equal(t, float64(i), decoded[i]["sequence_number"])

		// Verify that every payload is recognized by the pinned official SDK's
		// Responses API stream-event union.
		var official responses.ResponseStreamEventUnion
		require.NoError(t, json.Unmarshal(event.data, &official))
		assert.Equal(t, event.eventType, official.Type)
	}
	return decoded
}

func eventTypes(events []streamEvent) []string {
	types := make([]string, len(events))
	for i, event := range events {
		types[i] = event.eventType
	}
	return types
}

func TestBuildResponseStreamAssistantLifecycle(t *testing.T) {
	response := responseFixture(t, `[{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hello, 世界 👋","annotations":[]}]}]`)
	original, err := json.Marshal(response)
	require.NoError(t, err)

	events, err := buildResponseStream(response)
	require.NoError(t, err)
	assert.Equal(t, []string{
		"response.created",
		"response.in_progress",
		"response.output_item.added",
		"response.content_part.added",
		"response.output_text.delta",
		"response.output_text.done",
		"response.content_part.done",
		"response.output_item.done",
		"response.completed",
	}, eventTypes(events))
	decoded := decodeStreamEvents(t, events)

	for _, i := range []int{0, 1} {
		snapshot := decoded[i]["response"].(map[string]any)
		assert.Equal(t, "in_progress", snapshot["status"])
		assert.Empty(t, snapshot["output"])
	}

	added := decoded[2]["item"].(map[string]any)
	assert.Equal(t, "msg_1", added["id"])
	assert.Equal(t, "in_progress", added["status"])
	assert.Empty(t, added["content"])

	for _, i := range []int{3, 4, 5, 6} {
		assert.Equal(t, "msg_1", decoded[i]["item_id"])
		assert.Equal(t, float64(0), decoded[i]["output_index"])
		assert.Equal(t, float64(0), decoded[i]["content_index"])
	}
	assert.Equal(t, "Hello, 世界 👋", decoded[4]["delta"])
	assert.Equal(t, "Hello, 世界 👋", decoded[5]["text"])

	done := decoded[7]["item"].(map[string]any)
	assert.Equal(t, "completed", done["status"])
	require.Len(t, done["content"], 1)
	assert.Equal(t, "Hello, 世界 👋", done["content"].([]any)[0].(map[string]any)["text"])

	completed := decoded[8]["response"].(map[string]any)
	assert.Equal(t, "completed", completed["status"])
	assert.Len(t, completed["output"], 1)
	var originalWire map[string]any
	require.NoError(t, json.Unmarshal(original, &originalWire))
	assert.Equal(t, originalWire["output"], completed["output"], "completed response must retain the fixture's final output")

	after, err := json.Marshal(response)
	require.NoError(t, err)
	assert.JSONEq(t, string(original), string(after), "building snapshots must not mutate the fixture")
}

func TestBuildResponseStreamFunctionCallLifecycle(t *testing.T) {
	response := responseFixture(t, `[{"id":"fc_1","type":"function_call","status":"completed","call_id":"call_1","namespace":"mcp__tools","name":"calculate","arguments":"{\"n\":2}"}]`)
	events, err := buildResponseStream(response)
	require.NoError(t, err)
	assert.Equal(t, []string{
		"response.created",
		"response.in_progress",
		"response.output_item.added",
		"response.function_call_arguments.delta",
		"response.function_call_arguments.done",
		"response.output_item.done",
		"response.completed",
	}, eventTypes(events))
	decoded := decodeStreamEvents(t, events)

	added := decoded[2]["item"].(map[string]any)
	assert.Equal(t, "in_progress", added["status"])
	assert.Equal(t, "", added["arguments"])
	assert.Equal(t, "mcp__tools", added["namespace"])
	assert.Equal(t, "fc_1", decoded[3]["item_id"])
	assert.Equal(t, float64(0), decoded[3]["output_index"])
	assert.Equal(t, `{"n":2}`, decoded[3]["delta"])
	assert.NotContains(t, decoded[3], "arguments")
	assert.Equal(t, `{"n":2}`, decoded[4]["arguments"])
	assert.Equal(t, "calculate", decoded[4]["name"])
	assert.Equal(t, "fc_1", decoded[4]["item_id"])
	assert.Equal(t, float64(0), decoded[4]["output_index"])
	done := decoded[5]["item"].(map[string]any)
	assert.Equal(t, "completed", done["status"])
	assert.Equal(t, "mcp__tools", done["namespace"])

	for _, event := range events {
		assert.NotContains(t, event.eventType, "function_call_output")
	}
}

func TestBuildResponseStreamToolSearchCall(t *testing.T) {
	response := responseFixture(t, `[{"id":"ts_1","type":"tool_search_call","status":"completed","call_id":"search_1","execution":"client","arguments":{"query":"mcp__tools add_numbers","limit":10}}]`)
	events, err := buildResponseStream(response)
	require.NoError(t, err)
	assert.Equal(t, []string{
		"response.created",
		"response.in_progress",
		"response.output_item.done",
		"response.completed",
	}, eventTypes(events))

	var done map[string]any
	require.NoError(t, json.Unmarshal(events[2].data, &done))
	assert.Equal(t, float64(2), done["sequence_number"])
	item := done["item"].(map[string]any)
	assert.Equal(t, "tool_search_call", item["type"])
	assert.Equal(t, "ts_1", item["id"])
	assert.Equal(t, "search_1", item["call_id"])
	assert.Equal(t, "client", item["execution"])
	assert.Equal(t, "mcp__tools add_numbers", item["arguments"].(map[string]any)["query"])
}

func TestBuildResponseStreamEmptyText(t *testing.T) {
	response := responseFixture(t, `[{"id":"msg_empty","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"","annotations":[]}]}]`)
	events, err := buildResponseStream(response)
	require.NoError(t, err)
	decoded := decodeStreamEvents(t, events)
	assert.Contains(t, decoded[4], "delta")
	assert.Equal(t, "", decoded[4]["delta"])
	assert.Contains(t, decoded[5], "text")
	assert.Equal(t, "", decoded[5]["text"])
}

func TestBuildResponseStreamMultipleOutputsAndContent(t *testing.T) {
	response := responseFixture(t, `[
		{"id":"msg_0","type":"message","status":"completed","role":"assistant","content":[
			{"type":"output_text","text":"zero","annotations":[]},
			{"type":"output_text","text":"one","annotations":[]}
		]},
		{"id":"fc_1","type":"function_call","status":"completed","call_id":"call_1","name":"tool","arguments":"{}"},
		{"id":"msg_2","type":"message","status":"completed","role":"assistant","content":[
			{"type":"output_text","text":"two","annotations":[]}
		]}
	]`)
	events, err := buildResponseStream(response)
	require.NoError(t, err)
	decoded := decodeStreamEvents(t, events)

	activeItems := map[string]bool{}
	activeParts := map[string]bool{}
	seenDeltaCoordinates := [][2]int{}
	for _, event := range decoded {
		typ := event["type"].(string)
		switch typ {
		case "response.output_item.added":
			activeItems[event["item"].(map[string]any)["id"].(string)] = true
		case "response.content_part.added":
			key := event["item_id"].(string) + ":" + eventNumber(event, "content_index")
			assert.True(t, activeItems[event["item_id"].(string)])
			activeParts[key] = true
		case "response.output_text.delta":
			key := event["item_id"].(string) + ":" + eventNumber(event, "content_index")
			assert.True(t, activeItems[event["item_id"].(string)], "delta requires an active output item")
			assert.True(t, activeParts[key], "delta requires an active content part")
			seenDeltaCoordinates = append(seenDeltaCoordinates, [2]int{int(event["output_index"].(float64)), int(event["content_index"].(float64))})
		case "response.content_part.done":
			key := event["item_id"].(string) + ":" + eventNumber(event, "content_index")
			delete(activeParts, key)
		case "response.output_item.done":
			delete(activeItems, event["item"].(map[string]any)["id"].(string))
		}
	}
	assert.Equal(t, [][2]int{{0, 0}, {0, 1}, {2, 0}}, seenDeltaCoordinates)
	assert.Empty(t, activeItems)
	assert.Empty(t, activeParts)
}

func eventNumber(event map[string]any, key string) string {
	data, _ := json.Marshal(event[key])
	return string(data)
}

func TestStreamingHandlerSSEFramingAndValidation(t *testing.T) {
	valid := responseFixture(t, `[{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"ok","annotations":[]}]}]`)
	recorder := httptest.NewRecorder()
	provider := NewOpenAIResponseProvider(nil)
	provider.handleResponsesStreamingResponse(recorder, valid)

	result := recorder.Result()
	assert.Equal(t, http.StatusOK, result.StatusCode)
	assert.Equal(t, "text/event-stream", result.Header.Get("Content-Type"))
	body := recorder.Body.String()
	assert.Contains(t, body, "event: response.created\ndata: {")
	assert.True(t, strings.HasSuffix(body, "data: [DONE]\n\n"))

	malformed := responseFixture(t, `[{"type":"message","status":"completed","role":"assistant","content":[]}]`)
	recorder = httptest.NewRecorder()
	provider.handleResponsesStreamingResponse(recorder, malformed)
	assert.Equal(t, http.StatusInternalServerError, recorder.Code)
	assert.Contains(t, recorder.Body.String(), "output[0] message id is required")
	assert.NotEqual(t, "text/event-stream", recorder.Header().Get("Content-Type"), "validation must happen before SSE headers")
}

func TestBuildResponseStreamRejectsUnsupportedOutput(t *testing.T) {
	response := responseFixture(t, `[{"type":"reasoning","id":"reason_1","status":"completed","summary":[]}]`)
	_, err := buildResponseStream(response)
	require.Error(t, err)
	assert.Contains(t, err.Error(), `type "reasoning" is unsupported for streaming`)

	response = responseFixture(t, `[{"type":"function_call_output","call_id":"call_1","output":"done"}]`)
	_, err = buildResponseStream(response)
	require.Error(t, err)
	assert.Contains(t, err.Error(), `type "function_call_output" is unsupported for streaming`)
}
