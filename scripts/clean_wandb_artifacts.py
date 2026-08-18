#!/usr/bin/env python3
"""
Prune WandB model artifact versions that do not have any alias (i.e. are untagged),
keeping those with aliases like `latest` or your custom alias (e.g. `best`).
"""

import time
import wandb
from typing import Optional, Set

def prune_untagged_models(
    project: str,
    entity: str,
    keep_aliases: Optional[Set[str]] = None,
    dry_run: bool = False,
    sleep_secs: float = 0.5,
    verbose: bool = False,
):
    verbose = verbose or dry_run
    
    if keep_aliases is None:
        keep_aliases = {"latest", "best"}

    api = wandb.Api(overrides={"project": project, "entity": entity})
    # iterate through artifact collections of type "model"
    for art_type in api.artifact_types(project=project):
        if art_type.type != "model":
            continue
        for coll in art_type.collections():
            coll_name = coll.name
            print(f"Collection: {coll_name}")

            try:
                versions = api.artifacts(type_name="model", name=coll_name)
            except Exception as e:
                print(f"  [WARN] Could not list versions for collection {coll_name}: {e}")
                continue

            for version in versions:
                try:
                    aliases = set(version.aliases)
                except Exception as e:
                    print(f"  [WARN] Could not fetch aliases for version {version.name}: {e}")
                    continue

                if not aliases:
                    print(f"  Would delete (no alias): {version.name}") if verbose else None
                    if not dry_run:
                        try:
                            version.delete()
                            print(f"    Deleted {version.name}") if verbose else None
                        except Exception as e:
                            print(f"Error!    Failed to delete {version.name}: {e}") if verbose else None
                else:
                    print(f"  Keeping (has aliases {aliases}): {version.name}") if verbose else None

                if not dry_run:
                    time.sleep(sleep_secs)

    print("Pruning finished.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prune untagged WandB model artifacts")
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Only print actions, don’t delete")
    parser.add_argument("--keep", nargs="*", default=["latest", "best"],
                        help="Aliases to always keep")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable verbose logging")
    args = parser.parse_args()

    prune_untagged_models(
        project=args.project,
        entity=args.entity,
        keep_aliases=set(args.keep),
        dry_run=args.dry_run,
    )
