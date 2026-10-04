mod dto;
mod error;
mod routes;
mod sse;

use std::future::Future;

use axum::Router;
use tokio::net::TcpListener;

pub use crate::server::error::ServerError;
use crate::worker::{Worker, WorkerOptions, spawn};

pub struct ServerOptions {
    pub worker: WorkerOptions,
    pub bind: String,
}

pub fn serve(options: ServerOptions) -> Result<(), ServerError> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(ServerError::Runtime)?
        .block_on(start(options))
}

pub async fn serve_with(
    listener: TcpListener,
    worker: Worker,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), ServerError> {
    let served = axum::serve(listener, router(&worker))
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(ServerError::Http);
    tracing::info!(message = "HTTP server stopped.");
    let joined = worker.join().await.map_err(ServerError::from);
    served.and(joined)
}

pub fn router(worker: &Worker) -> Router {
    routes::build(worker.handle(), worker.info().clone())
}

async fn start(options: ServerOptions) -> Result<(), ServerError> {
    let listener = TcpListener::bind(&options.bind)
        .await
        .map_err(ServerError::Bind)?;
    let worker = spawn(options.worker).await?;
    tracing::info!(message = "OpenAI-compatible server listening.", bind = %options.bind);
    serve_with(listener, worker, shutdown_signal()).await
}

async fn shutdown_signal() {
    match tokio::signal::ctrl_c().await {
        Ok(()) => tracing::info!(message = "Shutdown requested."),
        Err(error) => {
            tracing::error!(message = "Cannot listen for the shutdown signal.", %error);
            std::future::pending::<()>().await;
        }
    }
}
