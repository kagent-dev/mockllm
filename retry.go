package mockllm

import (
	"context"
	"time"
)

// RetryWithBackoff executes the given function with retries on an error, up to the given number
// of attempts with exponential backoff baseDelay*(2^attempt) up to maxDelay
func RetryWithBackoff(
	ctx context.Context,
	attempts int,
	baseDelay time.Duration,
	maxDelay time.Duration,
	f func() error,
) error {
	var err error
	for i := 0; i < attempts; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			if err = f(); err == nil {
				return nil
			}
			// exponential backoff
			delay := baseDelay * (1 << uint(i))
			if delay > maxDelay {
				delay = maxDelay
			}
			time.Sleep(delay)
		}
	}
	return err
}
