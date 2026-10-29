-- Supabase setup for the nonprofit legal-data dashboard.
-- Run this in Supabase SQL Editor before connecting the dashboard.
-- Row-level security is enabled and no public/anon policies are created.
-- IMPORTANT: The current dashboard connector uses the anon key without Supabase Auth.
-- Do not add permissive anon policies for real/private documents. Add Auth sign-in and
-- owner_id-based policies before enabling cloud sync for sensitive data.

create table if not exists public.dashboard_workspaces (
  workspace_id text primary key,
  payload jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now()
);

alter table public.dashboard_workspaces enable row level security;

comment on table public.dashboard_workspaces is
'Workspace JSON for nonprofit governance analysis dashboard. Keep private; configure authenticated owner policies before production use.';
