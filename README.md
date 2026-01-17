# Mistral Agents Integration Pipe for Open WebUI

Enable native Mistral AI Agents support in Open WebUI through this custom pipe. This pipe allows you to use agents created in the Mistral Console directly within your Open WebUI instance.

## Features

- ✅ Direct integration with Mistral AI Agents API (`/v1/agents/completions`)
- ✅ Support for streaming responses
- ✅ Easy configuration through Valves
- ✅ Works with any agent created in Mistral Console
- ✅ HTTP request retries with exponential backoff
- ✅ Support for Mistral beta conversations API
- ✅ File handling and parameter support

## Requirements

- Mistral AI API Key (get it at https://console.mistral.ai/)
- An Agent ID from Mistral Console (starts with `ag:`)
- Open WebUI instance

## Installation

1. **Install this pipe** in your Open WebUI instance
2. **Click the gear icon** (⚙️) to configure Valves
3. **Set the following values**:
   - `MISTRAL_API_KEY`: Your Mistral API key
   - `AGENT_ID`: Your agent ID from Mistral Console (e.g., `ag:abc123...`)
4. **Save** and you're ready to go!

## How to Create an Agent

1. Go to https://console.mistral.ai/build/agents
2. Create and configure your agent (add tools, instructions, knowledge base, etc.)
3. Copy the Agent ID (starts with `ag:`)
4. Paste it in the `AGENT_ID` valve

## Usage

Once configured, select the "Mistral Agent" model from your model selector and start chatting! The pipe will automatically route your requests to your configured agent.

## Configuration Options

### Required Valves

- `MISTRAL_API_KEY`: Your Mistral AI API key
- `AGENT_ID`: Your agent ID from Mistral Console

### Optional Valves

- `MAX_RETRIES`: Maximum number of retries for HTTP requests (default: 3)
- `RETRY_DELAY`: Initial delay between retries in seconds (default: 1)
- `STREAMING_ENABLED`: Enable or disable streaming responses (default: true)

## Important Notes

- **Agent ID vs Model**: This pipe uses the Mistral Agents API endpoint (`/v1/agents/completions`), which is different from the standard chat completions endpoint
- **Streaming**: Supports streaming responses for real-time interaction
- **Tools**: Your agent's configured tools and functions will work automatically
- **Retries**: HTTP requests are automatically retried with exponential backoff

## Troubleshooting

- **403 Error**: Check that your API key is valid
- **Agent not found**: Verify the Agent ID is correct and starts with `ag:`
- **No response**: Ensure your agent is properly configured in Mistral Console
- **Connection issues**: Check your network and retry settings

## Development

### Building

To build this project, ensure you have Python installed and run:

```bash
python mistral-agents-openwebui-pipe.py
```

### Testing

You can test the pipe by setting the required environment variables and running the script:

```bash
export MISTRAL_API_KEY="your_api_key"
export AGENT_ID="your_agent_id"
python mistral-agents-openwebui-pipe.py
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Commit your changes (`git commit -am 'feat: add your feature'`)
5. Push to the branch (`git push origin feature/your-feature`)
6. Create a new Pull Request

## Credits

Created for the Open WebUI community to enable seamless Mistral Agents integration.

## License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.
