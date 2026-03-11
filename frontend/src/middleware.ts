import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

// The access_token cookie is scoped to api.agentsights.org (the backend domain).
// Browser requests to the frontend domain never include that cookie, so server-side
// middleware cannot check it. Auth protection is handled client-side in
// dashboard/layout.tsx via hydrateFromServer() → /users/me/ → redirect on 401.
export function middleware(_request: NextRequest) {
  return NextResponse.next();
}

export const config = {
  matcher: [],
};
