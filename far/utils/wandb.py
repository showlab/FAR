from dataclasses import dataclass
from typing import List, Optional


@dataclass
class wandb_config:
    project: str = 'far'  # wandb project name
    entity: Optional[str] = None  # wandb entity name
    job_type: Optional[str] = None
    tags: Optional[List[str]] = None
    group: Optional[str] = None
    notes: Optional[str] = None
    mode: Optional[str] = None
    name: Optional[str] = None
    dir: Optional[str] = None
    group: Optional[str] = None
    resume: str = 'allow'
    id: Optional[str] = None
