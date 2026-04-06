"""Networking and process coordination (tlspyo, sockets, timeouts)."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class DistributedConfig(BaseModel):
    """Ports, socket buffers, and timeouts for trainer / worker / server processes."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    server_port: Annotated[int, Field(ge=1, le=65535)] = Field(
        default=55555,
        description="Primary TCP port the central TMRL server listens on.",
    )
    local_port_server: Annotated[int, Field(ge=1, le=65535)] = Field(
        default=55556,
        description="Auxiliary local bind port used by the server process on the host.",
    )
    local_port_trainer: Annotated[int, Field(ge=1, le=65535)] = Field(
        default=55557,
        description="Local port the trainer binds for inbound connections.",
    )
    local_port_worker: Annotated[int, Field(ge=1, le=65535)] = Field(
        default=55558,
        description="Local port workers bind when registering with the server.",
    )
    buffer_size: PositiveInt = Field(
        default=536_870_912,
        description="SO_RCVBUF/SNDBUF-style buffer size in bytes for bulk tensor streaming.",
    )
    header_size: PositiveInt = Field(
        default=12,
        description="Fixed wire-protocol header size in bytes prepended to each message.",
    )
    socket_timeout_connect_trainer: Annotated[float, Field(gt=0)] = Field(
        default=300.0,
        description="Seconds to wait when the trainer opens an outbound connection.",
    )
    socket_timeout_accept_trainer: Annotated[float, Field(gt=0)] = Field(
        default=300.0,
        description="Seconds the server waits to accept an incoming trainer connection.",
    )
    socket_timeout_connect_rollout: Annotated[float, Field(gt=0)] = Field(
        default=300.0,
        description="Seconds to wait when a rollout worker connects to the server.",
    )
    socket_timeout_accept_rollout: Annotated[float, Field(gt=0)] = Field(
        default=300.0,
        description="Seconds the server waits to accept a worker connection.",
    )
    socket_timeout_communicate: Annotated[float, Field(gt=0)] = Field(
        default=30.0,
        description="Per-read/write timeout during active message exchange.",
    )
    select_timeout_outbound: Annotated[float, Field(gt=0)] = Field(
        default=30.0,
        description="Timeout for select/poll on outbound queues in the I/O loop.",
    )
    ack_timeout_worker_to_server: Annotated[float, Field(gt=0)] = Field(
        default=300.0,
        description="Maximum wait for application-level ACK from server after worker send.",
    )
    ack_timeout_trainer_to_server: Annotated[float, Field(gt=0)] = Field(
        default=300.0,
        description="Maximum wait for ACK from server after trainer control messages.",
    )
    ack_timeout_server_to_worker: Annotated[float, Field(gt=0)] = Field(
        default=300.0,
        description="Maximum wait for worker to acknowledge server directives.",
    )
    ack_timeout_server_to_trainer: Annotated[float, Field(gt=0)] = Field(
        default=7200.0,
        description="Long-timeout path for large trainer↔server transfers (e.g. weights).",
    )
    recv_timeout_trainer_from_server: Annotated[float, Field(gt=0)] = Field(
        default=7200.0,
        description="Blocking recv timeout when the trainer waits on payloads from the server.",
    )
    recv_timeout_worker_from_server: Annotated[float, Field(gt=0)] = Field(
        default=600.0,
        description="Blocking recv timeout when workers wait on server payloads.",
    )
    wait_before_reconnection: Annotated[float, Field(ge=0)] = Field(
        default=10.0,
        description="Sleep duration before retrying after a dropped connection.",
    )
    loop_sleep_time: Annotated[float, Field(ge=0)] = Field(
        default=1.0,
        description="Idle sleep in dispatcher loops to reduce busy-wait CPU usage.",
    )
    password: str = Field(
        default="change_me",
        description="Shared secret for tlspyo; override with env TMRL_PASSWORD in production.",
    )
    use_tls: bool = Field(
        default=False,
        description="When true, negotiate TLS on tlspyo transports (requires credentials).",
    )
    tls_hostname: str = Field(
        default="default",
        description="Server name indication / certificate hostname for TLS.",
    )
    tls_credentials_directory: str = Field(
        default="",
        description="Directory containing TLS material; empty uses non-custom credential loading.",
    )
    public_ip_server: str = Field(
        default="0.0.0.0",
        description="Address workers/trainers use when not on localhost (advertised server IP).",
    )
    localhost_worker: bool = Field(
        default=True,
        description="If true, workers default to 127.0.0.1 instead of public_ip_server.",
    )
    localhost_trainer: bool = Field(
        default=True,
        description="If true, trainer defaults to 127.0.0.1 instead of public_ip_server.",
    )
    num_workers: int = Field(
        default=-1,
        description="Requested rollout worker count; negative means auto / unlimited pool.",
    )
