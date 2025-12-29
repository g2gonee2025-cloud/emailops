import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
<<<<<<< HEAD
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
=======
import { QueryProvider } from './lib/queryClient'
>>>>>>> cc32a600 (feat(frontend): Add QueryClient and QueryProvider)
import './index.css'
import App from './App.tsx'
import { AuthProvider } from './contexts/AuthContext.tsx'
import { ToastProvider } from './components/ui/Toast.tsx'

const queryClient = new QueryClient()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
<<<<<<< HEAD
      <QueryClientProvider client={queryClient}>
        <ToastProvider>
          <AuthProvider>
            <App />
          </AuthProvider>
        </ToastProvider>
      </QueryClientProvider>
=======
      <QueryProvider>
        <App />
      </QueryProvider>
>>>>>>> cc32a600 (feat(frontend): Add QueryClient and QueryProvider)
    </BrowserRouter>
  </StrictMode>,
)
