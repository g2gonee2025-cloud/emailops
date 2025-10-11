# Gemini Superpowers Guide

Welcome to your supercharged Gemini environment! This guide will walk you through the new features and how to use them to boost your productivity.

## Table of Contents

1.  [Personas](#personas)
2.  [Instructions](#instructions)
3.  [MCP Guide](#mcp-guide)
4.  [Workflows](#workflows)

---

## Personas

You now have access to a set of predefined personas that you can use to guide Gemini's behavior. These personas are located in the `.gemini/personas` directory.

### Available Personas:

-   **`senior_software_engineer.md`**: A seasoned expert in software architecture, design patterns, and clean code.
-   **`devops_expert.md`**: A specialist in cloud infrastructure, CI/CD, and automation.

### How to Use Personas:

To use a persona, simply copy and paste its content at the beginning of your prompt. For example:

```
[Paste content of senior_software_engineer.md here]

Please review the following code...
```

---

## Instructions

Reusable instructions for common tasks are available in the `.gemini/instructions` directory.

### Available Instructions:

-   **`generate_unit_tests.md`**: A template for generating comprehensive unit tests.
-   **`refactor_for_performance.md`**: A template for refactoring code to improve performance.

### How to Use Instructions:

Similar to personas, you can copy and paste the content of an instruction into your prompt. You can also combine a persona and an instruction for more specific requests.

---

## MCP Guide

The `.gemini/MCP_GUIDE.md` file provides a curated list of recommended MCP servers that you can use to extend Gemini's capabilities. Refer to this guide to learn how to connect to and use these external tools.

---

## Workflows

The `.gemini/workflows` directory contains PowerShell scripts that automate common tasks.

### Available Workflows:

-   **`code_review.ps1`**: Automates the process of preparing a code review request.
-   **`generate_tests.ps1`**: Automates the process of preparing a unit test generation request.

### How to Use Workflows:

1.  Open a PowerShell terminal.
2.  Navigate to the `.gemini/workflows` directory.
3.  Run the desired script with the path to the target file as an argument.

**Example:**

```powershell
.\code_review.ps1 -FilePath "C:\path\to\your\file.py"
```

The script will output a `gcloud` command that you can copy and paste into your terminal to interact with the Gemini CLI. **Note:** You will need to replace `your-gcp-project-id` with your actual GCP project ID.
