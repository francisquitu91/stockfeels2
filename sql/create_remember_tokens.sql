-- Create table remember_tokens and enable row-level security with policy

-- 1) Create the preferred table to store refresh tokens
create table if not exists public.remember_tokens (
  id uuid primary key references auth.users(id) on delete cascade,
  refresh_token text not null,
  inserted_at timestamptz default now()
);

-- 2) Enable Row Level Security on the table
alter table public.remember_tokens enable row level security;

-- 3) Create a policy that allows authenticated users to manage only their own token
create policy manage_own_token on public.remember_tokens
  for all
  using ( auth.uid() = id )
  with check ( auth.uid() = id );

-- Optional: grant minimal privileges to anon role if you want public-readable access (not recommended)
-- grant select on public.remember_tokens to anon;

-- Quick test upsert example (run as an admin or via SQL editor):
-- upsert into public.remember_tokens (id, refresh_token) values ('00000000-0000-0000-0000-000000000000', 'example-token')
-- on conflict (id) do update set refresh_token = excluded.refresh_token;
