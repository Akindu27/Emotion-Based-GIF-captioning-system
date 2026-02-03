import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

const mock = {
  from: (_: string) => ({
    insert: async () => ({ data: { id: 'mock-id' }, error: null }),
    select: async () => ({ data: null, error: null }),
    single: async () => ({ data: { id: 'mock-id' }, error: null }),
    update: async () => ({ data: null, error: null }),
    eq: async () => ({ data: null, error: null }),
  }),
} as any;

export const supabase = (() => {
  if (!supabaseUrl || !supabaseAnonKey) {
    console.warn('VITE_SUPABASE_URL or VITE_SUPABASE_ANON_KEY not set — using mock supabase client');
    return mock;
  }

  return createClient(supabaseUrl, supabaseAnonKey);
})();
