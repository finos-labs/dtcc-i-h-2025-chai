# dtcc-i-h-2025-chai

This repository is a private project under FINOS Labs. It primarily uses TypeScript and includes a Next.js-based frontend located in the `dtcc-frontend` directory.

## Project Overview

**dtcc-i-h-2025-chai** is the codebase for a modern web application, leveraging the Next.js framework for rapid development and scalability. The project is currently under active development and is intended for internal or partner use.

## Features

- Built with Next.js (React 19)
- Uses Radix UI components and Tailwind CSS for modern, accessible UI
- Integrates with Amazon Bedrock for advanced AI and LLM capabilities
- Includes an MCP (Main Control Plane) backend for core business logic and orchestration
- Features RAG (Retrieval-Augmented Generation) functionality for enhanced information retrieval and generation
- Integrates with OpenAI and Groq SDKs
- Utilities for PDF parsing and charting

## Prerequisites

- Node.js (version 18 or later recommended)
- npm or yarn

## Installation (Frontend)

1. Clone the repository:
   ```sh
   git clone https://github.com/finos-labs/dtcc-i-h-2025-chai.git
   cd dtcc-i-h-2025-chai/dtcc-frontend
   ```

2. Install dependencies:
   ```sh
   npm install
   ```
   or
   ```sh
   yarn install
   ```

## Launch Instructions (Frontend)

### Development Mode

To run the app locally in development mode (with hot reload):

```sh
npm run dev
```
or
```sh
yarn dev
```

### Production Mode

To build and start the app for production:

```sh
npm run build
npm start
```
or
```sh
yarn build
yarn start
```

## Linting

To check for lint errors:

```sh
npm run lint
```
or
```sh
yarn lint
```

---

## Hosting Backend Services: MCP and Bedrock Instances

The project also requires backend services for full functionality. Specifically, you must run both the **MCP** (Main Control Plane) and **Bedrock** instances alongside the frontend.

### MCP (Main Control Plane)

- The MCP service coordinates and manages core business logic and backend integration.
- Make sure the MCP service is running and accessible to the frontend before launching the application.
- Refer to the `mcp/` directory (or relevant documentation) for setup instructions, dependencies, and how to start the MCP instance.

### Bedrock

- Bedrock provides foundational data services, advanced AI/LLM APIs, and RAG support required by the app.
- This service must also be running and accessible.
- Refer to the `bedrock/` directory (or relevant documentation) for detailed setup and launch instructions.

**Note:**  
Adjust environment variables or configuration files as needed to point the frontend to your running MCP and Bedrock services. Check `.env` files in each directory for configuration options.

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Contributing

Please contact the repository admins or FINOS Labs if you are interested in contributing.

---

© FINOS Labs 2025. All rights reserved.
