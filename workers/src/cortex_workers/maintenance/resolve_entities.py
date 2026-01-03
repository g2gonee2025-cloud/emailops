import difflib
import logging
from collections import defaultdict

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

    def _get_candidates(
        self, session: Session, node_type: str, with_email: bool = False
    ) -> list[dict]:
        """Fetch entity nodes and return them as a list of candidate dicts."""
        query = select(EntityNode).where(EntityNode.type == node_type)
        if with_email:
            query = query.where(EntityNode.properties.has_key("email"))
        nodes = session.scalars(query).all()
        return [
            {
                "id": n.node_id,
                "name": n.name,
                "props": n.properties or {},
                "len": len(n.name),
            }
            for n in nodes
        ]

    def _score_candidate(self, c: dict) -> int:
        """Score a candidate to determine which one to keep in a merge."""
        s = c["len"]
        if c["props"]:
            s += 10
        return s

    def _process_and_merge_groups(
        self, session: Session, groups: list[list[dict]], dry_run: bool
    ) -> int:
        """
        For a list of groups of candidates, pick a canonical and merge others.
        A group is a list of candidate dicts.
        """
        merged_count = 0
        merges_to_execute = []
        for group in groups:
            if len(group) > 1:
                keep_cand = max(group, key=self._score_candidate)
                for cand in group:
                    if cand["id"] != keep_cand["id"]:
                        merges_to_execute.append((keep_cand, cand))
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

    def _group_similar_candidates(self, candidates: list[dict]) -> list[list[dict]]:
        """Group candidates by fuzzy name matching."""
        sorted_candidates = sorted(candidates, key=lambda x: x["len"], reverse=True)
        processed_ids = set()
        groups = []
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
                groups.append(group)
        return groups

    def _resolve_by_fuzzy_name(self, session: Session, dry_run: bool) -> int:
        """Merge nodes with high name similarity (>92%)."""
        logger.info("Resolving by Fuzzy Name...")
        merged_count = 0
        for node_type in ["PERSON", "ORGANIZATION"]:
            candidates = self._get_candidates(session, node_type)
            if len(candidates) > 1000:
                logger.warning(
                    f"Skipping fuzzy match for {node_type} - too many nodes ({len(candidates)})"
                )
                continue
            groups = self._group_similar_candidates(candidates)
            merged_count += self._process_and_merge_groups(session, groups, dry_run)
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
        candidates = self._get_candidates(session, "PERSON", with_email=True)
        email_map: dict[str, list] = defaultdict(list)
        for c in candidates:
            email = c["props"].get("email")
            if email:
                email_map[email.lower().strip()].append(c)
        groups = [group for group in email_map.values() if len(group) > 1]
        return self._process_and_merge_groups(session, groups, dry_run)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    resolver = EntityResolver()
    resolver.run()
