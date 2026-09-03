"""Provider catalog and credential capture for the dashboard.

The dashboard needs three things the analysis pipeline does not: which
providers it can offer, which models each one serves (and what reasoning
efforts those models accept), and a credential it can hand to the runner
without the user pasting a secret into a text box.

Credentials are resolved from the tooling the user has already configured —
``~/.aws`` for Bedrock, ``~/.databrickscfg`` for Databricks — and are returned
as environment variables so they never appear in a command line, where ``ps``
would expose them to every process on the machine.
"""

from __future__ import annotations

import configparser
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

CATALOG_PATH = Path(__file__).resolve().parents[2] / "config" / "models.yaml"
DATABRICKS_CONFIG = Path.home() / ".databrickscfg"
# The env var the openai_compatible client reads its key from; see
# tradingagents/llm_clients/api_key_env.py.
COMPATIBLE_KEY_ENV = "OPENAI_COMPATIBLE_API_KEY"


class CredentialError(RuntimeError):
    """Raised when a provider is selected but its credentials are missing."""


@dataclass(frozen=True)
class Model:
    id: str
    label: str
    efforts: tuple[str, ...] = ()


@dataclass(frozen=True)
class Provider:
    name: str
    label: str
    auth: str
    llm_provider: str
    models: tuple[Model, ...]
    profile: str | None = None
    default_quick: str | None = None
    default_deep: str | None = None

    def index_of(self, model_id: str | None) -> int:
        """Position of ``model_id`` in the model list, or 0 when unknown."""
        return next(
            (i for i, m in enumerate(self.models) if m.id == model_id), 0
        )

    def model(self, model_id: str) -> Model | None:
        return next((m for m in self.models if m.id == model_id), None)

    def efforts_for(self, model_id: str) -> tuple[str, ...]:
        found = self.model(model_id)
        return found.efforts if found else ()


def load_catalog(path: Path | None = None) -> dict[str, Provider]:
    """Read the editable YAML catalog into provider records."""
    raw = yaml.safe_load((path or CATALOG_PATH).read_text()) or {}
    catalog: dict[str, Provider] = {}
    for name, spec in raw.items():
        catalog[name] = Provider(
            name=name,
            label=spec.get("label", name),
            auth=spec.get("auth", ""),
            llm_provider=spec.get("llm_provider", name),
            profile=spec.get("profile"),
            default_quick=spec.get("default_quick"),
            default_deep=spec.get("default_deep"),
            models=tuple(
                Model(
                    id=m["id"],
                    label=m.get("label", m["id"]),
                    efforts=tuple(m.get("efforts", ())),
                )
                for m in spec.get("models", [])
            ),
        )
    return catalog


def _databricks_profile(profile: str) -> dict[str, str]:
    if not DATABRICKS_CONFIG.exists():
        return {}
    parser = configparser.ConfigParser()
    parser.read(DATABRICKS_CONFIG)
    # ``in`` rather than has_section: Databricks names its default profile
    # [DEFAULT], which configparser treats as a magic section that
    # has_section() deliberately excludes.
    return dict(parser[profile]) if profile in parser else {}


def _oauth_token(profile: str) -> str:
    """Mint a short-lived OAuth token via the CLI, for profiles without a PAT."""
    try:
        import json

        result = subprocess.run(
            ["databricks", "auth", "token", "-p", profile],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        return json.loads(result.stdout).get("access_token", "")
    except (OSError, subprocess.SubprocessError, ValueError):
        return ""


def databricks_credentials(profile: str = "DEFAULT") -> tuple[str, str]:
    """Return ``(host, token)`` for a Databricks profile, prompting for nothing.

    Resolution order follows the Databricks tooling convention: explicit
    environment variables win, then a personal access token in
    ``~/.databrickscfg``, then an OAuth token minted by the CLI.
    """
    section = _databricks_profile(profile)
    host = os.environ.get("DATABRICKS_HOST") or section.get("host", "")
    token = os.environ.get("DATABRICKS_TOKEN") or section.get("token", "")
    if host and not token:
        token = _oauth_token(profile)
    if not host or not token:
        raise CredentialError(
            f"No Databricks credentials for profile '{profile}'. "
            f"Run: databricks auth login --profile {profile}"
        )
    return host.rstrip("/"), token


def base_url(host: str) -> str:
    """The OpenAI-compatible endpoint Databricks serves models on."""
    return f"{host.rstrip('/')}/serving-endpoints"


def resolve(provider: Provider) -> tuple[str | None, dict[str, str]]:
    """Return ``(base_url, env)`` needed to run against ``provider``.

    The returned env is merged into the runner's environment by the caller, so
    secrets stay out of argv.
    """
    if provider.auth == "databricks-profile":
        host, token = databricks_credentials(provider.profile or "DEFAULT")
        return base_url(host), {COMPATIBLE_KEY_ENV: token}
    # Bedrock authenticates through the standard AWS credential chain, which the
    # runner already resolves from --aws-profile. Nothing to capture here.
    return None, {}
