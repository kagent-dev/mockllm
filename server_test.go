package mockllm_test

import (
	"bytes"
	"context"
	"embed"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/kagent-dev/mockllm"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

//go:embed testdata/*.json
var testdataFS embed.FS

// loadTestData loads a JSON file from testdata and unmarshals it into the target
func loadTestData(t *testing.T, filename string, target interface{}) {
	t.Helper()
	data, err := testdataFS.ReadFile("testdata/" + filename)
	require.NoError(t, err)
	err = json.Unmarshal(data, target)
	require.NoError(t, err)
}

// newOpenAIMock creates an OpenAIMock with the given parameters, handling marshal/unmarshal
func newOpenAIMock(t *testing.T, name string, matchType mockllm.MatchType, request openai.ChatCompletionNewParams, response openai.ChatCompletion) mockllm.OpenAIMock {
	t.Helper()
	mock := mockllm.OpenAIMock{
		Name:     name,
		Response: response,
		Match: mockllm.OpenAIRequestMatch{
			MatchType: matchType,
			Message:   request.Messages[len(request.Messages)-1],
		},
	}

	// Marshal and unmarshal the request to get it in the right format
	reqBytes, err := json.Marshal(request)
	require.NoError(t, err)
	err = json.Unmarshal(reqBytes, &mock.Match)
	require.NoError(t, err)

	return mock
}

// newOpenAIResponseMock creates an OpenAIResponseMock with the given parameters
func newOpenAIResponseMock(t *testing.T, name string, matchType mockllm.MatchType, request responses.ResponseNewParams, response responses.Response) mockllm.OpenAIResponseMock {
	t.Helper()
	mock := mockllm.OpenAIResponseMock{
		Name:     name,
		Response: response,
		Match: mockllm.OpenAIResponseRequestMatch{
			MatchType: matchType,
			Input:     request.Input,
		},
	}

	// Marshal and unmarshal to get proper structure
	reqBytes, err := json.Marshal(request)
	require.NoError(t, err)
	err = json.Unmarshal(reqBytes, &mock.Match)
	require.NoError(t, err)

	return mock
}

// newOpenAIEmbeddingMock creates an OpenAIEmbeddingMock with the given parameters
func newOpenAIEmbeddingMock(t *testing.T, name string, matchType mockllm.MatchType, request openai.EmbeddingNewParams, response openai.CreateEmbeddingResponse) mockllm.OpenAIEmbeddingMock {
	t.Helper()
	mock := mockllm.OpenAIEmbeddingMock{
		Name:     name,
		Response: response,
		Match: mockllm.OpenAIEmbeddingRequestMatch{
			MatchType: matchType,
			Input:     request.Input,
		},
	}

	// Marshal and unmarshal to get proper structure
	reqBytes, err := json.Marshal(request)
	require.NoError(t, err)
	err = json.Unmarshal(reqBytes, &mock.Match)
	require.NoError(t, err)

	return mock
}

// newAnthropicMock creates an AnthropicMock with the given parameters
func newAnthropicMock(t *testing.T, name string, matchType mockllm.MatchType, request anthropic.MessageNewParams, response anthropic.Message) mockllm.AnthropicMock {
	t.Helper()
	mock := mockllm.AnthropicMock{
		Name:     name,
		Response: response,
		Match: mockllm.AnthropicRequestMatch{
			MatchType: matchType,
			Message:   request.Messages[len(request.Messages)-1],
		},
	}

	// Marshal and unmarshal the request to get it in the right format
	reqBytes, err := json.Marshal(request)
	require.NoError(t, err)
	err = json.Unmarshal(reqBytes, &mock.Match)
	require.NoError(t, err)

	return mock
}

// startTestServer starts a test server and returns the base URL and cleanup function
func startTestServer(t *testing.T, config mockllm.Config) (string, func()) {
	t.Helper()
	server := mockllm.NewServer(config)
	baseURL, err := server.Start(t.Context())
	require.NoError(t, err)
	return baseURL, func() {
		server.Stop(context.Background())
	}
}

func TestSimpleAnthropicMock(t *testing.T) {
	anthropicRequest := anthropic.MessageNewParams{
		Model:     "claude-3-5-sonnet-20240620",
		MaxTokens: 1000,
		Messages: []anthropic.MessageParam{
			{
				Role: anthropic.MessageParamRoleUser,
				Content: []anthropic.ContentBlockParamUnion{
					{
						OfText: &anthropic.TextBlockParam{
							Text: "Hello",
						},
					},
				},
			},
		},
	}

	var anthropicResponse anthropic.Message
	loadTestData(t, "anthropic_mock.json", &anthropicResponse)

	mock := newAnthropicMock(t, "test-response", mockllm.MatchTypeContains, anthropicRequest, anthropicResponse)
	config := mockllm.Config{
		Anthropic: []mockllm.AnthropicMock{mock},
	}

	baseURL, cleanup := startTestServer(t, config)
	defer cleanup()

	reqBytes, err := json.Marshal(anthropicRequest)
	require.NoError(t, err)

	req, err := http.NewRequest("POST", baseURL+"/v1/messages", bytes.NewReader(reqBytes))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", "test-key")
	req.Header.Set("anthropic-version", "2023-06-01")

	client := &http.Client{}
	resp, err := client.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close() //nolint:errcheck

	// Check response
	assert.Equal(t, http.StatusOK, resp.StatusCode)

	var responseBody map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&responseBody)
	require.NoError(t, err)

	assert.Equal(t, "msg_123", responseBody["id"])
	assert.Equal(t, "message", responseBody["type"])
}

func TestHealthCheck(t *testing.T) {
	config := mockllm.Config{
		OpenAI:         []mockllm.OpenAIMock{},
		OpenAIResponse: []mockllm.OpenAIResponseMock{},
		Anthropic:      []mockllm.AnthropicMock{},
	}
	baseURL, cleanup := startTestServer(t, config)
	defer cleanup()

	resp, err := http.Get(baseURL + "/health")
	require.NoError(t, err)
	defer resp.Body.Close() //nolint:errcheck

	assert.Equal(t, http.StatusOK, resp.StatusCode)

	var responseBody map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&responseBody)
	require.NoError(t, err)

	assert.Equal(t, "healthy", responseBody["status"])
	assert.Equal(t, "mock-llm", responseBody["service"])
	assert.NotNil(t, responseBody["openai"])
	assert.NotNil(t, responseBody["openai_response"])
	assert.NotNil(t, responseBody["anthropic"])
}

func TestLoadConfig(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.json")

	data, err := testdataFS.ReadFile("testdata/example_config.json")
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(configPath, data, 0o600))

	config, err := mockllm.LoadConfig(configPath)
	require.NoError(t, err)

	assert.Equal(t, "127.0.0.1:0", config.ListenAddr)
	require.Len(t, config.OpenAI, 1)
	assert.Equal(t, "hello", config.OpenAI[0].Name)
}

func TestOpenAICompletionMock(t *testing.T) {
	// Setup simple response for non-streaming
	simpleRequest := openai.ChatCompletionNewParams{
		Model: "gpt-4o-mini",
		Messages: []openai.ChatCompletionMessageParamUnion{
			{
				OfUser: &openai.ChatCompletionUserMessageParam{
					Role: "user",
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfString: openai.String("Hello"),
					},
				},
			},
		},
	}

	var simpleResponse openai.ChatCompletion
	loadTestData(t, "openai_mock.json", &simpleResponse)
	simpleMock := newOpenAIMock(t, "test-response", mockllm.MatchTypeExact, simpleRequest, simpleResponse)

	// Setup tool call response for streaming
	toolCallRequest := openai.ChatCompletionNewParams{
		Model: "gpt-4.1-mini",
		Messages: []openai.ChatCompletionMessageParamUnion{
			{
				OfUser: &openai.ChatCompletionUserMessageParam{
					Role: "user",
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfString: openai.String("What is 2+2?"),
					},
				},
			},
		},
	}

	var toolCallResponse openai.ChatCompletion
	loadTestData(t, "openai_function_mock.json", &toolCallResponse)
	toolCallMock := newOpenAIMock(t, "calculate_request", mockllm.MatchTypeContains, toolCallRequest, toolCallResponse)

	config := mockllm.Config{
		OpenAI: []mockllm.OpenAIMock{simpleMock, toolCallMock},
	}

	baseURL, cleanup := startTestServer(t, config)
	defer cleanup()

	client := openai.NewClient(
		option.WithBaseURL(baseURL+"/v1/"),
		option.WithAPIKey("test-key"),
	)

	// Test non-streaming response
	t.Run("non-streaming", func(t *testing.T) {
		resp, err := client.Chat.Completions.New(t.Context(), simpleRequest)
		require.NoError(t, err)

		assert.Equal(t, "chatcmpl-123", resp.ID)
		assert.Equal(t, "chat.completion", string(resp.Object))
		assert.Equal(t, "Hello! How can I help you today?", resp.Choices[0].Message.Content)
	})

	// Test streaming response (tool calls)
	t.Run("streaming", func(t *testing.T) {
		stream := client.Chat.Completions.NewStreaming(t.Context(), toolCallRequest)

		var receivedToolCalls []openai.ChatCompletionChunkChoiceDeltaToolCall

		for stream.Next() {
			evt := stream.Current()
			if len(evt.Choices) > 0 {
				if len(evt.Choices[0].Delta.ToolCalls) > 0 {
					receivedToolCalls = append(receivedToolCalls, evt.Choices[0].Delta.ToolCalls...)
				}
			}
		}

		if err := stream.Err(); err != nil {
			t.Fatalf("Stream error: %v", err)
		}

		require.Len(t, receivedToolCalls, 1)
		assert.Equal(t, "calculate", receivedToolCalls[0].Function.Name)
		assert.Equal(t, `{"expression": "2+2"}`, receivedToolCalls[0].Function.Arguments)
	})
}

func TestOpenAIResponseMock(t *testing.T) {
	// Setup haiku response for non-streaming
	haikuRequest := responses.ResponseNewParams{
		Model: openai.ChatModelGPT4,
		Input: responses.ResponseNewParamsInputUnion{
			OfString: openai.String("Write me a haiku about Kagent"),
		},
	}

	var haikuResponse responses.Response
	loadTestData(t, "openai_response_mock.json", &haikuResponse)
	haikuMock := newOpenAIResponseMock(t, "haiku-response", mockllm.MatchTypeContains, haikuRequest, haikuResponse)

	// The first tool-round-trip request returns only the model's function call.
	funcRequest := responses.ResponseNewParams{
		Model: openai.ChatModelGPT4,
		Input: responses.ResponseNewParamsInputUnion{
			OfString: openai.String("Calculate 2+2"),
		},
	}

	var funcResponse responses.Response
	loadTestData(t, "openai_response_function_mock.json", &funcResponse)
	funcMock := newOpenAIResponseMock(t, "function-response", mockllm.MatchTypeContains, funcRequest, funcResponse)

	// The follow-up request is selected by the function_call_output in the last
	// input-list position, not by the original prompt earlier in the list.
	funcOutputRequest := responses.ResponseNewParams{
		Model: openai.ChatModelGPT4,
		Input: responses.ResponseNewParamsInputUnion{
			OfString: openai.String("call_func_123"),
		},
	}
	funcOutputMock := newOpenAIResponseMock(t, "function-output-response", mockllm.MatchTypeContains, funcOutputRequest, haikuResponse)

	// Setup both mocks on the same server and use request matching to determine which response to return
	config := mockllm.Config{
		OpenAIResponse: []mockllm.OpenAIResponseMock{haikuMock, funcMock, funcOutputMock},
	}

	baseURL, cleanup := startTestServer(t, config)
	defer cleanup()

	client := openai.NewClient(
		option.WithBaseURL(baseURL+"/v1/"),
		option.WithAPIKey("test-key"),
	)

	// Test non-streaming response (haiku)
	t.Run("non-streaming", func(t *testing.T) {
		resp, err := client.Responses.New(t.Context(), haikuRequest)
		require.NoError(t, err)

		assert.Equal(t, "resp_123", resp.ID)
		outputText := resp.OutputText()
		assert.Contains(t, outputText, "Kagent finds its mind")
	})

	t.Run("tool round trip matches only last list item", func(t *testing.T) {
		first, err := client.Responses.New(t.Context(), funcRequest)
		require.NoError(t, err)
		require.Len(t, first.Output, 1)
		assert.Equal(t, "function_call", first.Output[0].Type)

		followUp := responses.ResponseNewParams{
			Model: openai.ChatModelGPT4,
			Input: responses.ResponseNewParamsInputUnion{
				OfInputItemList: responses.ResponseInputParam{
					responses.ResponseInputItemParamOfMessage("Calculate 2+2", responses.EasyInputMessageRoleUser),
					responses.ResponseInputItemParamOfFunctionCall("{\"expression\":\"2+2\"}", "call_func_123", "calculate"),
					responses.ResponseInputItemParamOfFunctionCallOutput("call_func_123", "4"),
				},
			},
		}
		second, err := client.Responses.New(t.Context(), followUp)
		require.NoError(t, err)
		assert.Equal(t, "resp_123", second.ID, "the original prompt must not select the first mock again")
		assert.Contains(t, second.OutputText(), "Kagent finds its mind")
	})
}

func TestOpenAICompletionAndResponseMocks(t *testing.T) {
	// Chat Completions mock
	openaiRequest := openai.ChatCompletionNewParams{
		Model: "gpt-4o-mini",
		Messages: []openai.ChatCompletionMessageParamUnion{
			{
				OfUser: &openai.ChatCompletionUserMessageParam{
					Role: "user",
					Content: openai.ChatCompletionUserMessageParamContentUnion{
						OfString: openai.String("Hello"),
					},
				},
			},
		},
	}

	var openaiResponse openai.ChatCompletion
	loadTestData(t, "openai_mock.json", &openaiResponse)
	chatMock := newOpenAIMock(t, "chat-test", mockllm.MatchTypeExact, openaiRequest, openaiResponse)

	// Responses API mock
	responseRequest := responses.ResponseNewParams{
		Model: openai.ChatModelGPT4,
		Input: responses.ResponseNewParamsInputUnion{
			OfString: openai.String("Write me a haiku about Kagent"),
		},
	}

	var response responses.Response
	loadTestData(t, "openai_response_mock.json", &response)
	responseMock := newOpenAIResponseMock(t, "response-test", mockllm.MatchTypeContains, responseRequest, response)

	config := mockllm.Config{
		OpenAI:         []mockllm.OpenAIMock{chatMock},
		OpenAIResponse: []mockllm.OpenAIResponseMock{responseMock},
	}

	baseURL, cleanup := startTestServer(t, config)
	defer cleanup()

	client := openai.NewClient(
		option.WithBaseURL(baseURL+"/v1/"),
		option.WithAPIKey("test-key"),
	)

	chatResp, err := client.Chat.Completions.New(t.Context(), openaiRequest)
	require.NoError(t, err)
	assert.Equal(t, "chatcmpl-123", chatResp.ID)
	assert.Contains(t, chatResp.Choices[0].Message.Content, "Hello")

	// Test that string version mock matches message list version request
	// This simulates how some frameworks send message lists instead of strings
	// Sends {"input": [{"role": "user", "content": "..."}]}
	responseRequestTest := responses.ResponseNewParams{
		Model: openai.ChatModelGPT4,
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: responses.ResponseInputParam{
				responses.ResponseInputItemParamOfMessage(
					"Write me a haiku about Kagent",
					responses.EasyInputMessageRoleUser,
				),
			},
		},
	}

	responseResp, err := client.Responses.New(t.Context(), responseRequestTest)
	require.NoError(t, err)
	assert.Equal(t, "resp_123", responseResp.ID)
	assert.Contains(t, responseResp.OutputText(), "Kagent finds its mind")
}

func TestOpenAIEmbeddingMock(t *testing.T) {
	embeddingRequestStr := openai.EmbeddingNewParams{
		Model: openai.EmbeddingModel("text-embedding-3-small"),
		Input: openai.EmbeddingNewParamsInputUnion{
			OfString: openai.String("Hello world"),
		},
	}

	var embeddingResponseStr openai.CreateEmbeddingResponse
	loadTestData(t, "openai_embedding_mock.json", &embeddingResponseStr)
	embeddingMockStr := newOpenAIEmbeddingMock(t, "embedding-str-test", mockllm.MatchTypeContains, embeddingRequestStr, embeddingResponseStr)

	// 2. Test array of strings input (batch generation)
	embeddingRequestList := openai.EmbeddingNewParams{
		Model: openai.EmbeddingModel("text-embedding-3-small"),
		Input: openai.EmbeddingNewParamsInputUnion{
			OfArrayOfStrings: []string{"Hello world", "Test embeddings"},
		},
		Dimensions: openai.Int(512),
	}

	var embeddingResponseList openai.CreateEmbeddingResponse
	loadTestData(t, "openai_embedding_mock.json", &embeddingResponseList)
	embeddingMockList := newOpenAIEmbeddingMock(t, "embedding-list-test", mockllm.MatchTypeContains, embeddingRequestList, embeddingResponseList)

	config := mockllm.Config{
		OpenAIEmbeddings: []mockllm.OpenAIEmbeddingMock{embeddingMockStr, embeddingMockList},
	}

	baseURL, cleanup := startTestServer(t, config)
	defer cleanup()

	client := openai.NewClient(
		option.WithBaseURL(baseURL+"/v1/"),
		option.WithAPIKey("test-key"),
	)

	t.Run("single string input", func(t *testing.T) {
		resp, err := client.Embeddings.New(t.Context(), embeddingRequestStr)
		require.NoError(t, err)

		assert.Equal(t, openai.EmbeddingModel("text-embedding-3-small"), resp.Model)
		assert.Equal(t, int64(0), resp.Data[0].Index)
		assert.Len(t, resp.Data[0].Embedding, 768)
	})

	t.Run("array of strings input with dimensions truncation", func(t *testing.T) {
		resp, err := client.Embeddings.New(t.Context(), embeddingRequestList)
		require.NoError(t, err)

		assert.Equal(t, openai.EmbeddingModel("text-embedding-3-small"), resp.Model)
		assert.Len(t, resp.Data, 2)
		assert.Equal(t, int64(0), resp.Data[0].Index)
		assert.Len(t, resp.Data[0].Embedding, 512)
		assert.Equal(t, int64(1), resp.Data[1].Index)
		assert.Len(t, resp.Data[1].Embedding, 512)
	})
}

func TestHeaderMatching(t *testing.T) {
	// Create two Anthropic mocks with the same body but different required headers
	anthropicRequest := anthropic.MessageNewParams{
		Model:     "claude-3-5-sonnet-20240620",
		MaxTokens: 1000,
		Messages: []anthropic.MessageParam{
			{
				Role: anthropic.MessageParamRoleUser,
				Content: []anthropic.ContentBlockParamUnion{
					{
						OfText: &anthropic.TextBlockParam{
							Text: "Hello",
						},
					},
				},
			},
		},
	}

	var responseTenantA anthropic.Message
	loadTestData(t, "anthropic_mock.json", &responseTenantA)
	// Override ID for tenant A
	responseTenantA.ID = "msg_tenant_a"

	var responseTenantB anthropic.Message
	loadTestData(t, "anthropic_mock.json", &responseTenantB)
	responseTenantB.ID = "msg_tenant_b"

	mockA := newAnthropicMock(t, "tenant-a", mockllm.MatchTypeContains, anthropicRequest, responseTenantA)
	mockA.Match.Headers = []mockllm.HeaderMatch{
		{Name: "X-Tenant-ID", Value: "tenant-a", MatchType: mockllm.MatchTypeExact},
	}

	mockB := newAnthropicMock(t, "tenant-b", mockllm.MatchTypeContains, anthropicRequest, responseTenantB)
	mockB.Match.Headers = []mockllm.HeaderMatch{
		{Name: "X-Tenant-ID", Value: "tenant-b", MatchType: mockllm.MatchTypeExact},
	}

	config := mockllm.Config{
		Anthropic: []mockllm.AnthropicMock{mockA, mockB},
	}

	baseURL, cleanup := startTestServer(t, config)
	defer cleanup()

	sendRequest := func(t *testing.T, tenantID string) *http.Response {
		t.Helper()
		reqBytes, err := json.Marshal(anthropicRequest)
		require.NoError(t, err)

		req, err := http.NewRequest("POST", baseURL+"/v1/messages", bytes.NewReader(reqBytes))
		require.NoError(t, err)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("x-api-key", "test-key")
		req.Header.Set("anthropic-version", "2023-06-01")
		req.Header.Set("X-Tenant-ID", tenantID)

		resp, err := http.DefaultClient.Do(req)
		require.NoError(t, err)
		return resp
	}

	t.Run("exact header match selects correct mock", func(t *testing.T) {
		resp := sendRequest(t, "tenant-a")
		defer resp.Body.Close()
		assert.Equal(t, http.StatusOK, resp.StatusCode)

		var body map[string]interface{}
		err := json.NewDecoder(resp.Body).Decode(&body)
		require.NoError(t, err)
		assert.Equal(t, "msg_tenant_a", body["id"])
	})

	t.Run("exact header match selects other mock", func(t *testing.T) {
		resp := sendRequest(t, "tenant-b")
		defer resp.Body.Close()
		assert.Equal(t, http.StatusOK, resp.StatusCode)

		var body map[string]interface{}
		err := json.NewDecoder(resp.Body).Decode(&body)
		require.NoError(t, err)
		assert.Equal(t, "msg_tenant_b", body["id"])
	})

	t.Run("header mismatch returns 404", func(t *testing.T) {
		resp := sendRequest(t, "tenant-unknown")
		defer resp.Body.Close()
		assert.Equal(t, http.StatusNotFound, resp.StatusCode)
	})
}

func TestHeaderContainsMatch(t *testing.T) {
	anthropicRequest := anthropic.MessageNewParams{
		Model:     "claude-3-5-sonnet-20240620",
		MaxTokens: 1000,
		Messages: []anthropic.MessageParam{
			{
				Role: anthropic.MessageParamRoleUser,
				Content: []anthropic.ContentBlockParamUnion{
					{
						OfText: &anthropic.TextBlockParam{
							Text: "Hello",
						},
					},
				},
			},
		},
	}

	var anthropicResponse anthropic.Message
	loadTestData(t, "anthropic_mock.json", &anthropicResponse)

	mock := newAnthropicMock(t, "contains-header", mockllm.MatchTypeContains, anthropicRequest, anthropicResponse)
	mock.Match.Headers = []mockllm.HeaderMatch{
		{Name: "Authorization", Value: "Bearer", MatchType: mockllm.MatchTypeContains},
	}

	config := mockllm.Config{
		Anthropic: []mockllm.AnthropicMock{mock},
	}

	baseURL, cleanup := startTestServer(t, config)
	defer cleanup()

	reqBytes, err := json.Marshal(anthropicRequest)
	require.NoError(t, err)

	t.Run("contains match succeeds", func(t *testing.T) {
		req, err := http.NewRequest("POST", baseURL+"/v1/messages", bytes.NewReader(reqBytes))
		require.NoError(t, err)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("x-api-key", "test-key")
		req.Header.Set("anthropic-version", "2023-06-01")
		req.Header.Set("Authorization", "Bearer sk-test-12345")

		resp, err := http.DefaultClient.Do(req)
		require.NoError(t, err)
		defer resp.Body.Close()
		assert.Equal(t, http.StatusOK, resp.StatusCode)
	})

	t.Run("contains match fails when header missing", func(t *testing.T) {
		req, err := http.NewRequest("POST", baseURL+"/v1/messages", bytes.NewReader(reqBytes))
		require.NoError(t, err)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("x-api-key", "test-key")
		req.Header.Set("anthropic-version", "2023-06-01")

		resp, err := http.DefaultClient.Do(req)
		require.NoError(t, err)
		defer resp.Body.Close()
		assert.Equal(t, http.StatusNotFound, resp.StatusCode)
	})
}
