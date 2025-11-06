# AI Agent Instructions


<!-- AI-COMMANDS:START -->
## Custom AI Command Workflows

This project has access to reusable AI command workflows from your dotfiles.

### How to Execute Commands

When the user requests any of these workflows, use the Bash tool to retrieve the command instructions:

```bash
ai-commands get <command-name>
```

The command will output the complete workflow instructions. Read the output carefully and follow all instructions exactly as written.

### Available Commands

- `create_plan`
- `implement_plan`
- `research_codebase`
- `save-session` - Create a detailed summary of the conversation and save it to ./sessions/
- `validate_plan`


### Usage Example

When user says "save the session" or "create a summary":
1. Run: `ai-commands get save-session`
2. Read the complete output
3. Follow all instructions in the returned content exactly

When user says "create a plan":
1. Run: `ai-commands get create-plan`
2. Follow the returned workflow instructions

### Command Location

All commands are stored in: `~/dotfiles/claude/commands/`

You can also run `ai-commands list` to see all available commands.

<!-- AI-COMMANDS:END -->
