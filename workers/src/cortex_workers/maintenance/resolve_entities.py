import difflib
import logging

from cortex.config.loader import get_config
from cortex.db.models import EntityNode
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger("cortex.workers.maintenance.resolve_entities")


class EntityResolver:
    def __init__(self):
        config = get_config()
        self.engine = create_engine(config.database.url)
        self.Session = sessionmaker(bind=self.engine)

    def run(self, dry_run: bool = False):
        """Run full entity resolution pipeline."""
        with self.Session() as session:
            logger.info("Starting Entity Resolution...")
            merged_email = self._resolve_by_email(session, dry_run)
            merged_fuzzy = self._resolve_by_fuzzy_name(session, dry_run)

            logger.info(
                f"Resolution Complete. Merged by Email: {merged_email}, Merged by Fuzzy Name: {merged_fuzzy}"
            )

    def _resolve_by_fuzzy_name(self, session: Session, dry_run: bool) -> int:
        """Merge nodes with high name similarity (>92%)."""
        logger.info("Resolving by Fuzzy Name...")
        merged_count = 0

        for node_type in ["PERSON", "ORGANIZATION"]:
            # Fetch all nodes first
            nodes = session.scalars(
                select(EntityNode).where(EntityNode.type == node_type)
            ).all()

            if len(nodes) > 1000:
                logger.warning(
                    f"Skipping fuzzy match for {node_type} - too many nodes ({len(nodes)})"
                )
                continue

            # Extract data to immutable structures to avoid ORM expiration issues
            # We need: id, name, properties (for canonical scoring)
            candidates = []
            for n in nodes:
                candidates.append(
                    {
                        "id": n.node_id,
                        "name": n.name,
                        "props": n.properties or {},
                        "len": len(n.name),
                    }
                )

            # Sort by name length (longest first)
            sorted_candidates = sorted(candidates, key=lambda x: x["len"], reverse=True)

            processed_ids = set()
            merges_to_execute = []

            for i, c1 in enumerate(sorted_candidates):
                if c1["id"] in processed_ids:
                    continue

                group = [c1]
                for c2 in sorted_candidates[i + 1 :]:
                    if c2["id"] in processed_ids:
                        continue

                    ratio = difflib.SequenceMatcher(
                        None, c1["name"].lower(), c2["name"].lower()
                    ).ratio()
                    if ratio > 0.92:
                        group.append(c2)
                        processed_ids.add(c2["id"])

                if len(group) > 1:
                    processed_ids.add(c1["id"])

                    # Pick canonical based on data
                    def score(c):
                        s = c["len"]
                        if c["props"]:
                            s += 10
                        return s

                    keep_cand = max(group, key=score)

                    for cand in group:
                        if cand["id"] != keep_cand["id"]:
                            merges_to_execute.append((keep_cand, cand))

            # Execute merges
            for keep, discard in merges_to_execute:
                self._merge_nodes_by_id(
                    session,
                    keep["id"],
                    keep["name"],
                    discard["id"],
                    discard["name"],
                    dry_run,
                )
                merged_count += 1

        return merged_count

    def _merge_nodes_by_id(
        self,
        session: Session,
        keep_id,
        keep_name,
        discard_id,
        discard_name,
        dry_run: bool,
    ):
        """Execute merge using IDs."""
        if dry_run:
            logger.info(
                f"[DRY RUN] Would merge '{discard_name}' ({discard_id}) INTO '{keep_name}' ({keep_id})"
            )
            return

        logger.info(f"Merging '{discard_name}' -> '{keep_name}'")
        try:
            session.execute(
                text("SELECT merge_entity_nodes(:keep, :discard)"),
                {"keep": keep_id, "discard": discard_id},
            )
            session.commit()
        except Exception as e:
            logger.error(f"Failed to merge {discard_name} -> {keep_name}: {e}")
            session.rollback()

    def _resolve_by_email(self, session: Session, dry_run: bool) -> int:
        """Merge PERSON nodes with identical email addresses."""
        logger.info("Resolving by Email...")
        # Fetch all PERSON nodes with email property
        nodes = session.scalars(
            select(EntityNode).where(
                EntityNode.type == "PERSON", EntityNode.properties.has_key("email")
            )
        ).all()

        # Group by email immediately to list of dicts/IDs to avoid ORM issues if we loop-commit
        email_map: dict[str, list] = {}
        for n in nodes:
            email = n.properties.get("email")
            if email:
                email = email.lower().strip()
                if email not in email_map:
                    email_map[email] = []
                # Store full object state or just ID? Using ID is safer if we refactor _merge_nodes
                # But _pick_canonical logic needs props.
                # Let's extract essential data
                email_map[email].append(
                    {
                        "id": n.node_id,
                        "name": n.name,
                        "props": n.properties or {},
                        "len": len(n.name),
                    }
                )

        merged_count = 0
        for _email, group in email_map.items():
            if len(group) > 1:
                # Pick canonical
                def score(c):
                    s = c["len"]
                    if c["props"]:
                        s += 10
                    return s

                keep_cand = max(group, key=score)

                for cand in group:
                    if cand["id"] != keep_cand["id"]:
                        self._merge_nodes_by_id(
                            session,
                            keep_cand["id"],
                            keep_cand["name"],
                            cand["id"],
                            cand["name"],
                            dry_run,
                        )
                        merged_count += 1
        return merged_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    resolver = EntityResolver()
    resolver.run()
