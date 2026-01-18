import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

export function middleware(request: NextRequest) {
  // Get token from cookies or check localStorage via headers
  const path = request.nextUrl.pathname;

  // For dashboard routes, check authentication
  if (path.startsWith("/dashboard")) {
    // Check if we have authentication info in the request
    // Since we can't access localStorage in middleware, we'll check cookies
    // The client-side auth store uses localStorage, so we need to handle this differently

    // For now, we'll let the client-side handle redirect from dashboard pages
    // The pages themselves will check auth and redirect if needed
    return NextResponse.next();
  }

  // If accessing login with token, redirect to dashboard
  // This will be handled client-side as well for consistency

  return NextResponse.next();
}

export const config = {
  matcher: ["/dashboard/:path*", "/login", "/signup"],
};
