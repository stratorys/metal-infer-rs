use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use thiserror::Error;

use crate::server::dto::ErrorBody;
use crate::worker::{SubmitError, WorkerError};

#[derive(Debug, Error)]
pub enum RequestError {
    #[error("invalid completion request")]
    InvalidJson(#[source] serde_json::Error),
    #[error("requested model is not loaded")]
    ModelNotLoaded,
    #[error("message content part type is not supported")]
    UnsupportedContentPart,
    #[error("max_tokens exceeds the context")]
    MaxTokensExceedContext,
    #[error("request queue is full")]
    QueueFull,
    #[error("inference worker stopped")]
    WorkerStopped,
    #[error(transparent)]
    Worker(#[from] WorkerError),
}

#[derive(Debug, Error)]
pub enum ServerError {
    #[error("cannot start the async runtime")]
    Runtime(#[source] std::io::Error),
    #[error("cannot bind the server address")]
    Bind(#[source] std::io::Error),
    #[error("HTTP server failed")]
    Http(#[source] std::io::Error),
    #[error(transparent)]
    Worker(#[from] WorkerError),
}

impl RequestError {
    pub const fn status(&self) -> StatusCode {
        match self {
            Self::InvalidJson(_)
            | Self::ModelNotLoaded
            | Self::UnsupportedContentPart
            | Self::MaxTokensExceedContext => StatusCode::BAD_REQUEST,
            Self::QueueFull => StatusCode::SERVICE_UNAVAILABLE,
            Self::WorkerStopped => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Worker(error) => {
                if error.is_client_error() {
                    StatusCode::BAD_REQUEST
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                }
            }
        }
    }
}

impl From<SubmitError> for RequestError {
    fn from(error: SubmitError) -> Self {
        match error {
            SubmitError::QueueFull => Self::QueueFull,
            SubmitError::Stopped => Self::WorkerStopped,
        }
    }
}

impl IntoResponse for RequestError {
    fn into_response(self) -> Response {
        match &self {
            Self::QueueFull | Self::WorkerStopped => {
                tracing::error!(message = "Completion request rejected.", error = %self);
            }
            Self::InvalidJson(_)
            | Self::ModelNotLoaded
            | Self::UnsupportedContentPart
            | Self::MaxTokensExceedContext
            | Self::Worker(_) => {}
        }
        (self.status(), Json(ErrorBody::from_error(&self))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;

    use super::RequestError;
    use crate::server::dto::ErrorBody;
    use crate::worker::WorkerError;

    #[test]
    fn statuses_follow_the_error_origin() {
        let cases = [
            (RequestError::ModelNotLoaded, StatusCode::BAD_REQUEST),
            (
                RequestError::UnsupportedContentPart,
                StatusCode::BAD_REQUEST,
            ),
            (
                RequestError::MaxTokensExceedContext,
                StatusCode::BAD_REQUEST,
            ),
            (RequestError::QueueFull, StatusCode::SERVICE_UNAVAILABLE),
            (
                RequestError::WorkerStopped,
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
            (
                RequestError::Worker(WorkerError::ContextExceeded),
                StatusCode::BAD_REQUEST,
            ),
            (
                RequestError::Worker(WorkerError::BatchFailed),
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
        ];
        cases.iter().for_each(|(error, status)| {
            assert_eq!(error.status(), *status, "unexpected status for {error:?}");
        });
    }

    #[test]
    fn server_errors_hide_their_message() {
        let body = serde_json::to_value(ErrorBody::from_error(&RequestError::Worker(
            WorkerError::BatchFailed,
        )))
        .expect("error body serializes");
        assert_eq!(
            body.pointer("/error/message"),
            Some(&serde_json::Value::from("internal server error")),
            "a server error must not expose its message"
        );
        assert_eq!(
            body.pointer("/error/type"),
            Some(&serde_json::Value::from("server_error")),
            "a server error must be typed server_error"
        );
    }

    #[test]
    fn client_errors_return_their_static_message() {
        let body = serde_json::to_value(ErrorBody::from_error(&RequestError::ModelNotLoaded))
            .expect("error body serializes");
        assert_eq!(
            body.pointer("/error/message"),
            Some(&serde_json::Value::from("requested model is not loaded")),
            "a client error must return its static message"
        );
        assert_eq!(
            body.pointer("/error/type"),
            Some(&serde_json::Value::from("invalid_request_error")),
            "a client error must be typed invalid_request_error"
        );
    }
}
