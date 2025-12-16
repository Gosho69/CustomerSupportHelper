import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

export function middleware(request: NextRequest) {
  // Temporarily disabled for demo purposes
  // const token = request.cookies.get('access_token')?.value;
  // const path = request.nextUrl.pathname;

  // // Check if accessing dashboard routes
  // if (path.startsWith('/dashboard')) {
  //   // If no token, redirect to login
  //   if (!token) {
  //     return NextResponse.redirect(new URL('/login', request.url));
  //   }
  // }

  // // If accessing login/signup with token, redirect to dashboard
  // if ((path === '/login' || path === '/signup') && token) {
  //   return NextResponse.redirect(new URL('/dashboard', request.url));
  // }

  return NextResponse.next();
}

export const config = {
  matcher: ["/dashboard/:path*", "/login", "/signup"],
};
