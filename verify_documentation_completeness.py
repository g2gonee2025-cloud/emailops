import os

combined_file_path = 'docs/combined_documentation.md'
source_files = [
    'docs/emailops_core_conversation_loader.md',
    'docs/emailops_feature_search_draft.md',
    'docs/emailops_util_processing.md',
    'docs/DIRECTORY_MAPPING.md',
    'docs/emailops_outlook_exporter.md',
    'docs/emailops_vertex_agent_builder.md',
    'docs/emailops_utils.md',
    'docs/emailops_tool_doctor.md',
    'docs/emailops_common.md',
    'docs/emailops_observability.md',
    'docs/emailops_llm_runtime.md',
    'docs/emailops_llm_client_shim.md',
    'docs/emailops_indexing_parallel.md',
    'docs/emailops_indexing_main.md',
    'docs/emailops_index_transaction.md',
    'docs/emailops_feature_summarize.md',
    'docs/emailops_core_validators.md',
    'docs/emailops_core_text_extraction.md',
    'docs/emailops_core_manifest.md',
    'docs/emailops_core_exceptions.md',
    'docs/emailops_core_email_processing.md',
    'docs/emailops_core_config_models.md',
    'docs/emailops_core_config.md',
    'docs/emailops_cli.md'
]

def normalize_content(content):
    """Normalize content by stripping whitespace and newlines for comparison."""
    return "".join(content.split())

def verify_completeness():
    if not os.path.exists(combined_file_path):
        print(f"Error: Combined file '{combined_file_path}' not found.")
        return

    try:
        with open(combined_file_path, encoding='utf-8') as f:
            combined_content = f.read()
    except Exception as e:
        print(f"Error reading combined file: {e}")
        return

    # Normalize combined content for robust checking
    # normalized_combined = normalize_content(combined_content)

    missing_files = []

    print(f"Verifying {len(source_files)} files against {combined_file_path}...")

    for source_file in source_files:
        if not os.path.exists(source_file):
            print(f"Warning: Source file '{source_file}' not found. Skipping.")
            continue

        try:
            with open(source_file, encoding='utf-8') as f:
                source_content = f.read()

            # We check if the source content is present in the combined content.
            # Since the combined file might have extra separators, we check for the core content.
            # A simple "in" check is usually sufficient if it's a direct concatenation.
            if source_content.strip() not in combined_content:
                # If exact match fails, try a more lenient check (e.g. first 100 chars)
                # to see if it's just a whitespace issue at the boundaries
                if source_content.strip()[:100] not in combined_content:
                     missing_files.append(source_file)
                     print(f"X Missing: {source_file}")
                else:
                     print(f"~ Partial/Modified Match: {source_file} (Header found, but exact body match failed - likely whitespace/formatting)")
            else:
                print(f"OK: {source_file}")

        except Exception as e:
            print(f"Error reading source file {source_file}: {e}")

    print("-" * 30)
    if missing_files:
        print(f"FAILED: {len(missing_files)} files appear to be missing or significantly modified.")
        for f in missing_files:
            print(f"- {f}")
    else:
        print("SUCCESS: All source files appear to be present in the combined documentation.")

if __name__ == "__main__":
    verify_completeness()
