import React, { Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import ProtectedLayout from './components/Layout';

// Helper for lazy loading views
const lazyView = (importFunc: () => Promise<{ default: React.ComponentType<any> }>) => {
  const LazyComponent = lazy(importFunc);
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <LazyComponent />
    </Suspense>
  );
};

// Lazy-loaded views
const LoginView = lazyView(() => import('./views/LoginView'));
const DashboardView = lazyView(() => import('./views/DashboardView'));
const AskView = lazyView(() => import('./views/AskView'));
const SearchView = lazyView(() => import('./views/SearchView'));
const ThreadView = lazyView(() => import('./views/ThreadView'));
const DraftView = lazyView(() => import('./views/DraftView'));
const IngestionView = lazyView(() => import('./views/IngestionView'));
const AdminDashboard = lazyView(() => import('./views/AdminDashboard'));

export const AppRoutes = () => (
  <Routes>
    <Route path="/login" element={LoginView} />
    <Route path="/" element={<ProtectedLayout />}>
      <Route index element={<Navigate to="/dashboard" replace />} />
      <Route path="dashboard" element={DashboardView} />
      <Route path="ask" element={AskView} />
      <Route path="search" element={SearchView} />
      <Route path="thread/:id" element={ThreadView} />
      <Route path="draft" element={DraftView} />
      <Route path="ingest" element={IngestionView} />
      <Route path="admin" element={AdminDashboard} />
    </Route>
  </Routes>
);
