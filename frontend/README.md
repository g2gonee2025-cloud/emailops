# Frontend

This directory contains the React-based frontend for the application.

## Development

To get started, install the dependencies and run the development server:

```bash
npm install
npm run dev
```

The application will be available at `http://localhost:5173`.

## Testing

To run the unit and component tests, use the following command:

```bash
npm run test
```

## Authentication

The frontend proxies API requests to the backend server, which is expected to be running on `http://localhost:8000`. Authentication is handled by the backend, and the frontend stores the authentication token in `localStorage`. No special environment variables are needed for the frontend to handle authentication.
