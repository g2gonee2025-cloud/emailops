#!/usr/bin/env python3
"""
Recommendations for implementing markdown schemas based on different use cases.
"""

# ===================================================================
# RECOMMENDATION MATRIX
# ===================================================================

RECOMMENDATIONS = {
    "high_structure_low_readability": {
        "approach": "current_json",
        "reason": "Keep current JSON schemas for maximum precision",
        "use_cases": [
            "API responses that need exact structure",
            "Database storage requirements",
            "Integration with strict downstream systems",
        ],
    },
    "high_structure_high_readability": {
        "approach": "yaml_frontmatter",
        "reason": "YAML frontmatter provides structure + readability",
        "use_cases": [
            "User-facing reports that need both data and narrative",
            "Documentation generation",
            "Human review workflows",
        ],
    },
    "medium_structure_high_readability": {
        "approach": "template_based",
        "reason": "Templates provide good structure while staying readable",
        "use_cases": [
            "Email drafts (your primary use case)",
            "Report generation",
            "Standardized communications",
        ],
    },
    "low_structure_high_readability": {
        "approach": "instructional_markdown",
        "reason": "Maximum flexibility and readability",
        "use_cases": [
            "Creative content generation",
            "Exploratory analysis",
            "Conversational interfaces",
        ],
    },
    "hybrid_requirements": {
        "approach": "json_embedded",
        "reason": "Preserves existing data structures while adding readability",
        "use_cases": [
            "Gradual migration from JSON schemas",
            "Mixed human/machine consumption",
            "Backward compatibility requirements",
        ],
    },
}


# ===================================================================
# SPECIFIC RECOMMENDATIONS FOR YOUR CODEBASE
# ===================================================================


def get_recommendation_for_schema(schema_name: str) -> dict:
    """Get specific recommendations for each schema in your codebase."""

    recommendations = {
        "email_draft_response": {
            "current_complexity": "high",
            "readability_importance": "high",
            "recommended_approach": "template_based",
            "reasoning": """
            Email drafts benefit greatly from readability for human review.
            The structure is consistent enough for templates.
            Citations table format is intuitive in markdown.
            """,
            "migration_effort": "medium",
            "example_benefit": "Reviewers can quickly scan citations and assumptions",
        },
        "thread_analysis_facts_ledger": {
            "current_complexity": "very_high",
            "readability_importance": "critical",
            "recommended_approach": "yaml_frontmatter",
            "reasoning": """
            Thread analysis is consumed by humans making business decisions.
            Complex nested structure (facts_ledger) suits YAML.
            Markdown content allows for detailed explanations.
            """,
            "migration_effort": "high",
            "example_benefit": "Executives can quickly identify risks and next actions",
        },
        "chat_response": {
            "current_complexity": "medium",
            "readability_importance": "high",
            "recommended_approach": "instructional_markdown",
            "reasoning": """
            Chat responses should feel natural and conversational.
            Citations can be embedded inline with markdown links.
            Less rigid structure is acceptable for chat.
            """,
            "migration_effort": "low",
            "example_benefit": "More natural conversation flow with inline citations",
        },
        "critic_feedback": {
            "current_complexity": "medium",
            "readability_importance": "medium",
            "recommended_approach": "current_json",
            "reasoning": """
            Critic feedback is primarily machine-consumed.
            Precise structure needed for programmatic processing.
            Readability is secondary to consistency.
            """,
            "migration_effort": "none",
            "example_benefit": "Keep current approach - no change needed",
        },
    }

    return recommendations.get(
        schema_name,
        {
            "recommended_approach": "evaluate_case_by_case",
            "reasoning": "Unknown schema - needs individual analysis",
        },
    )


# ===================================================================
# MIGRATION STRATEGY
# ===================================================================


def create_migration_plan() -> dict:
    """Create a phased migration plan for your codebase."""

    return {
        "phase_1_pilot": {
            "duration": "1-2 weeks",
            "scope": "Single schema (email_draft_response)",
            "approach": "template_based",
            "goals": [
                "Validate markdown template approach",
                "Test parsing reliability",
                "Measure user satisfaction improvement",
                "Benchmark performance impact",
            ],
            "success_criteria": [
                "90%+ parsing success rate",
                "Positive user feedback on readability",
                "No significant performance degradation",
            ],
        },
        "phase_2_expansion": {
            "duration": "2-3 weeks",
            "scope": "Thread analysis schema",
            "approach": "yaml_frontmatter",
            "goals": [
                "Handle complex nested structures",
                "Validate YAML parsing reliability",
                "Test with real business users",
            ],
            "success_criteria": [
                "Business users prefer new format",
                "Parsing handles all edge cases",
                "Integration tests pass",
            ],
        },
        "phase_3_optimization": {
            "duration": "1 week",
            "scope": "Chat responses",
            "approach": "instructional_markdown",
            "goals": [
                "Improve conversational flow",
                "Maintain citation accuracy",
                "Optimize for user experience",
            ],
        },
        "phase_4_hybrid": {
            "duration": "1 week",
            "scope": "Remaining schemas evaluation",
            "approach": "case_by_case",
            "goals": [
                "Evaluate each remaining schema",
                "Keep JSON where appropriate",
                "Document final architecture",
            ],
        },
    }


# ===================================================================
# IMPLEMENTATION GUIDANCE
# ===================================================================


def get_implementation_tips() -> dict:
    """Practical tips for implementing markdown schemas."""

    return {
        "parsing_robustness": [
            "Use multiple regex patterns for flexibility",
            "Implement fuzzy matching for section headers",
            "Add validation with meaningful error messages",
            "Consider using markdown parsing libraries like `markdown-it-py`",
        ],
        "llm_compliance": [
            "Test with multiple temperature settings",
            "Add examples in system prompts",
            "Use stop sequences to prevent format drift",
            "Implement retry logic with format corrections",
        ],
        "error_handling": [
            "Graceful degradation to partial parsing",
            "Fallback to original JSON approach on failures",
            "Detailed logging for parsing failures",
            "User feedback mechanisms for format issues",
        ],
        "testing_strategy": [
            "Unit tests for each parsing function",
            "Integration tests with real LLM outputs",
            "Property-based testing for edge cases",
            "A/B testing with users for readability",
        ],
    }


# ===================================================================
# CONCRETE NEXT STEPS
# ===================================================================

NEXT_STEPS = """
IMMEDIATE ACTIONS (This Week):

1. **Start with Email Draft Schema** (Highest ROI)
   - Implement template-based approach for email_draft_response  
   - Create robust parser with multiple fallback patterns
   - Test with existing context snippets
   - Measure user satisfaction vs current JSON

2. **Create Feature Flag System**
   - Add `use_markdown_schema` parameter to relevant functions
   - Allow A/B testing between JSON and Markdown approaches
   - Implement gradual rollout capability

3. **Enhance Error Handling** 
   - Modify complete_json() to accept markdown templates
   - Add markdown parsing with JSON fallback
   - Implement detailed error logging

SHORT TERM (Next 2-4 Weeks):

4. **Thread Analysis Migration**
   - Implement YAML frontmatter approach for facts_ledger
   - Create comprehensive parsing for nested structures
   - Test with business users for usability feedback

5. **Performance Benchmarking**
   - Compare token usage between JSON and Markdown
   - Measure parsing performance impact
   - Test LLM compliance rates

LONG TERM (1-2 Months):

6. **Hybrid Architecture**
   - Keep JSON for machine-only consumption (critic feedback)
   - Use Markdown for human-reviewed outputs
   - Document architectural decisions and patterns

7. **Tooling and Documentation**  
   - Create schema validation tools
   - Build markdown-to-JSON converters for compatibility
   - Update team documentation with best practices
"""

if __name__ == "__main__":
    print("=== MARKDOWN SCHEMA RECOMMENDATIONS ===")
    print(NEXT_STEPS)

    print("\n=== SCHEMA-SPECIFIC RECOMMENDATIONS ===")
    schemas = [
        "email_draft_response",
        "thread_analysis_facts_ledger",
        "chat_response",
        "critic_feedback",
    ]
    for schema in schemas:
        rec = get_recommendation_for_schema(schema)
        print(f"\n{schema}:")
        print(f"  Recommended: {rec.get('recommended_approach', 'N/A')}")
