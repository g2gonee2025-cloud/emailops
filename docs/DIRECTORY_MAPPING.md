# Directory Mapping

```
.
├── .editorconfig
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── Build-Conversations.fixed.ps1
├── check_gui_terminal_output.py
├── check_unused_code.py
├── configure_gemini_cli.ps1
├── create_test_structure.py
├── CURRENT_STATUS_AND_ISSUES.md
├── diagnose_body_issue.py
├── docker-compose.yml
├── Dockerfile
├── EmailOps.ps1
├── environment.yml
├── fix_issue_1_consolidate_strip_control_chars.py
├── list_outlook_folders.py
├── OUTLOOK_EXPORTER_REMEDIATION_COMPLETE.md
├── pyproject.toml
├── pytest.ini
├── requirements.txt
├── run_emailops.bat
├── run_gemini_cli.bat
├── run_outlook_export.py
├── sonar-project.properties
├── stop_and_fix.md
├── test_120b_reliability.py
├── test_all_models.py
├── test_cached_data_alignment.py
├── test_credential_validation.py
├── test_export_root.py
├── test_exporter_optimizations.py
├── test_fresh_export.py
├── test_gpt_120b.py
├── test_gradient_api.py
├── test_gui_restart.py
├── test_real_outlook_export.py
├── test_specific_folders.py
├── validate_critical_issues.py
├── VBA_CODE.VBA
├── verify_gui_services.py
├── .streamlit/
├── docs/
│   └── CANONICAL_BLUEPRINT.md
├── emailops/
│   ├── __init__.py
│   ├── cli.py
│   ├── common/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── types.py
│   ├── core_config_models.py
│   ├── core_config.py
│   ├── core_conversation_loader.py
│   ├── core_email_processing.py
│   ├── core_exceptions.py
│   ├── core_manifest.py
│   ├── core_text_extraction.py
│   ├── core_validators.py
│   ├── feature_search_draft.py
│   ├── feature_summarize.py
│   ├── gui/
│   │   ├── __init__.py
│   │   ├── .pre-commit-config.yaml
│   │   ├── analysis_panel.py
│   │   ├── app_fixed.py
│   │   ├── app.py
│   │   ├── base_panel.py
│   │   ├── chunking_panel.py
│   │   ├── CODE_STYLE.md
│   │   ├── components.py
│   │   ├── config_panel.py
│   │   ├── constants.py
│   │   ├── diagnose_services.py
│   │   ├── emailops_config.json
│   │   ├── file_panel.py
│   │   ├── launcher.py
│   │   ├── main_window.py
│   │   ├── pyproject.toml
│   │   ├── search_panel.py
│   │   ├── theme.py
│   │   └── visualization.py
│   ├── index_transaction.py
│   ├── indexing_main.py
│   ├── indexing_parallel.py
│   ├── llm_client_shim.py
│   ├── llm_runtime.py
│   ├── observability.py
│   ├── operations/
│   │   ├── __init__.py
│   │   ├── indexing.py
│   │   ├── search.py
│   │   └── summarize.py
│   ├── outlook_exporter/
│   │   ├── __init__.py
│   │   ├── attachments.py
│   │   ├── cli.py
│   │   ├── conversation.py
│   │   ├── exporter_v2.py
│   │   ├── exporter_v3.py
│   │   ├── exporter.py
│   │   ├── manifest_builder.py
│   │   ├── mapitags.py
│   │   ├── smtp_resolver.py
│   │   ├── state.py
│   │   └── utils.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── analysis_service.py
│   │   ├── atomic_file_service.py
│   │   ├── base_service.py
│   │   ├── batch_service.py
│   │   ├── chat_service.py
│   │   ├── chunking_service.py
│   │   ├── config_service.py
│   │   ├── email_service.py
│   │   ├── file_operations_validator.py
│   │   ├── file_service_secure.py
│   │   ├── file_service.py
│   │   ├── indexing_service.py
│   │   ├── resilience.py
│   │   └── search_service.py
│   ├── tool_doctor.py
│   ├── util_processing.py
│   ├── utils.py
│   └── vertex_agent_builder.py
├── emailops_docs/
│   ├── config.py.md
│   ├── conversation_loader.py.md
│   ├── doctor.py.corrected.md
│   ├── email_processing.py.md
│   ├── env_utils.py.md
│   ├── exceptions.py.md
│   ├── file_utils.py.md
│   ├── FINAL_IMPLEMENTATION_REPORT.md
│   ├── IMPLEMENTATION_VERIFICATION.md
│   ├── llm_client.py.corrected.md
│   ├── emailops_llm_runtime.md
│   ├── emailops_observability.md
│   ├── emailops_outlook_exporter.md
│   ├── emailops_tool_doctor.md
│   ├── processor.py.md
│   ├── README_VERTEX.md
│   ├── search_and_draft.py.corrected.md
│   ├── summarize_email_thread.py.corrected.md
│   ├── text_extraction.py.md
│   ├── utils.py.corrected.md
│   ├── validators.py.corrected.md
│   └── VERTEX_ALIGNMENT_SUMMARY.md
├── helpers & diagnostics/
│   ├── __init__.py
│   ├── analysis.py
│   ├── log/
│   ├── monitoring.py
│   ├── README.md
│   ├── setup/
│   │   ├── __init__.py
│   │   ├── enable_vertex_apis.py
│   │   └── verify_docker_wsl2.ps1
│   ├── setup.py
│   ├── sonarqube.py
│   ├── test_emailops_gui.py
│   └── testing.py
├── log/
├── secrets/
│   └── avp1-476017-e1ba80203362.json
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── debug/
│   ├── integration/
│   │   ├── test_core_workflows.py
│   │   ├── test_embedding_pipeline.py
│   │   └── test_security_integration.py
│   ├── search/
│   ├── SECURITY_TESTING.md
│   ├── test_all_accounts_live.py
│   ├── test_architectural_foundations.py
│   ├── test_config_integration.py
│   ├── test_fixes.py
│   ├── test_optimizations.py
│   ├── test_outlook_exporter.py
│   ├── test_result_validators.py
│   ├── test_search_and_draft.py
│   └── validation/
└── ui/
    └── __init__.py