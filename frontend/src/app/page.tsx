"use client";

import Link from "next/link";
import {
  ArrowRight,
  Mic,
  BarChart3,
  TrendingUp,
  Shield,
  Zap,
  Users,
  Menu,
  X,
} from "lucide-react";
import { useState } from "react";

export default function LandingPage() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-slate-900/80 backdrop-blur-md border-b border-white/10">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            {/* Logo */}
            <Link href="/" className="flex items-center space-x-2 group">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform duration-200">
                <Mic className="w-6 h-6 text-white" />
              </div>
              <span className="text-2xl font-bold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
                AgentSights
              </span>
            </Link>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center space-x-1">
              <a
                href="#features"
                className="px-4 py-2 text-gray-300 hover:text-white hover:bg-white/10 rounded-lg transition-all duration-200"
              >
                Features
              </a>
              <a
                href="#how-it-works"
                className="px-4 py-2 text-gray-300 hover:text-white hover:bg-white/10 rounded-lg transition-all duration-200"
              >
                How It Works
              </a>
              <a
                href="#benefits"
                className="px-4 py-2 text-gray-300 hover:text-white hover:bg-white/10 rounded-lg transition-all duration-200"
              >
                Benefits
              </a>
              <a
                href="#pricing"
                className="px-4 py-2 text-gray-300 hover:text-white hover:bg-white/10 rounded-lg transition-all duration-200"
              >
                Pricing
              </a>
            </div>

            {/* CTA Buttons */}
            <div className="hidden md:flex items-center space-x-3">
              <Link
                href="/login"
                className="px-5 py-2.5 bg-slate-800/50 backdrop-blur-sm border border-white/10 text-gray-300 hover:text-white hover:bg-slate-800 hover:border-white/20 transition-all duration-200 font-medium rounded-lg"
              >
                Sign In
              </Link>
              <Link
                href="/signup"
                className="px-6 py-2.5 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white rounded-lg transition-all duration-200 font-medium shadow-lg shadow-blue-500/30 hover:shadow-blue-500/50 hover:scale-105"
              >
                Get Started
              </Link>
            </div>

            {/* Mobile Menu Button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 text-gray-300 hover:text-white hover:bg-white/10 rounded-lg transition-all duration-200"
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? (
                <X className="w-6 h-6" />
              ) : (
                <Menu className="w-6 h-6" />
              )}
            </button>
          </div>

          {/* Mobile Menu */}
          {mobileMenuOpen && (
            <div className="md:hidden mt-4 pb-4 space-y-2 border-t border-white/10 pt-4">
              <a
                href="#features"
                onClick={() => setMobileMenuOpen(false)}
                className="block px-4 py-3 text-gray-300 hover:text-white hover:bg-white/10 rounded-lg transition-all duration-200"
              >
                Features
              </a>
              <a
                href="#how-it-works"
                onClick={() => setMobileMenuOpen(false)}
                className="block px-4 py-3 text-gray-300 hover:text-white hover:bg-white/10 rounded-lg transition-all duration-200"
              >
                How It Works
              </a>
              <a
                href="#benefits"
                onClick={() => setMobileMenuOpen(false)}
                className="block px-4 py-3 text-gray-300 hover:text-white hover:bg-white/10 rounded-lg transition-all duration-200"
              >
                Benefits
              </a>
              <a
                href="#pricing"
                onClick={() => setMobileMenuOpen(false)}
                className="block px-4 py-3 text-gray-300 hover:text-white hover:bg-white/10 rounded-lg transition-all duration-200"
              >
                Pricing
              </a>
              <div className="pt-4 space-y-2 border-t border-white/10 mt-4">
                <Link
                  href="/login"
                  onClick={() => setMobileMenuOpen(false)}
                  className="block px-4 py-3 text-center text-gray-300 hover:text-white hover:bg-white/10 rounded-lg transition-all duration-200 font-medium"
                >
                  Sign In
                </Link>
                <Link
                  href="/signup"
                  onClick={() => setMobileMenuOpen(false)}
                  className="block px-4 py-3 text-center bg-gradient-to-r from-blue-600 to-cyan-600 text-white rounded-lg font-medium"
                >
                  Get Started
                </Link>
              </div>
            </div>
          )}
        </div>
      </nav>

      {/* Hero Section */}
      <section className="container mx-auto px-6 py-20 text-center pt-32">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight">
            Transform Customer Support with
            <span className="bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
              {" "}
              AI-Powered Insights
            </span>
          </h1>
          <p className="text-xl text-gray-300 mb-10 leading-relaxed">
            Record, transcribe, and analyze every customer interaction. Get
            actionable insights, emotion detection, and AI-driven coaching to
            elevate your support team's performance.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/signup"
              className="px-8 py-4 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition font-semibold text-lg flex items-center justify-center group"
            >
              Start Free Trial
              <ArrowRight className="ml-2 group-hover:translate-x-1 transition" />
            </Link>
            <a
              href="#demo"
              className="px-8 py-4 bg-white/10 hover:bg-white/20 text-white rounded-lg transition font-semibold text-lg backdrop-blur-sm border border-white/20"
            >
              Watch Demo
            </a>
          </div>
          <p className="text-sm text-gray-400 mt-6">
            No credit card required • 14-day free trial
          </p>
        </div>

        {/* Hero Image/Dashboard Preview */}
        <div className="mt-16 max-w-5xl mx-auto">
          <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl shadow-2xl border border-white/10 p-8">
            <div className="bg-slate-950 rounded-lg overflow-hidden">
              <div className="flex items-center space-x-2 p-4 bg-slate-900 border-b border-slate-800">
                <div className="w-3 h-3 rounded-full bg-red-500"></div>
                <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
              </div>
              <div className="p-8 space-y-4">
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-blue-500/20 border border-blue-500/30 rounded-lg p-4">
                    <div className="text-blue-400 text-sm">Avg. Score</div>
                    <div className="text-3xl font-bold text-white">8.5/10</div>
                  </div>
                  <div className="bg-green-500/20 border border-green-500/30 rounded-lg p-4">
                    <div className="text-green-400 text-sm">Calls Analyzed</div>
                    <div className="text-3xl font-bold text-white">1,247</div>
                  </div>
                  <div className="bg-purple-500/20 border border-purple-500/30 rounded-lg p-4">
                    <div className="text-purple-400 text-sm">Sentiment</div>
                    <div className="text-3xl font-bold text-white">+12%</div>
                  </div>
                </div>
                <div className="bg-slate-900 rounded-lg p-6 border border-slate-800">
                  <div className="flex items-center justify-between mb-4">
                    <span className="text-gray-400">Recent Call Analysis</span>
                    <span className="text-blue-400 text-sm">Live</span>
                  </div>
                  <div className="space-y-2">
                    <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 w-3/4"></div>
                    </div>
                    <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-green-500 to-emerald-500 w-5/6"></div>
                    </div>
                    <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-purple-500 to-pink-500 w-2/3"></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="container mx-auto px-6 py-20">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Powerful Features for Modern Support Teams
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Everything you need to analyze, improve, and scale your customer
            support operations
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-8 hover:bg-white/10 transition">
            <div className="w-14 h-14 bg-blue-500/20 rounded-lg flex items-center justify-center mb-6">
              <Mic className="w-7 h-7 text-blue-400" />
            </div>
            <h3 className="text-2xl font-bold text-white mb-3">
              Smart Recording
            </h3>
            <p className="text-gray-400">
              Automatic call recording with dual-channel support. Capture every
              conversation with crystal-clear quality and metadata tracking.
            </p>
          </div>

          <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-8 hover:bg-white/10 transition">
            <div className="w-14 h-14 bg-purple-500/20 rounded-lg flex items-center justify-center mb-6">
              <Zap className="w-7 h-7 text-purple-400" />
            </div>
            <h3 className="text-2xl font-bold text-white mb-3">
              AI Transcription
            </h3>
            <p className="text-gray-400">
              Powered by Whisper AI for accurate transcription with speaker
              diarization, timestamps, and multi-language support.
            </p>
          </div>

          <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-8 hover:bg-white/10 transition">
            <div className="w-14 h-14 bg-green-500/20 rounded-lg flex items-center justify-center mb-6">
              <BarChart3 className="w-7 h-7 text-green-400" />
            </div>
            <h3 className="text-2xl font-bold text-white mb-3">
              Emotion Detection
            </h3>
            <p className="text-gray-400">
              Advanced sentiment analysis tracks emotions throughout calls -
              happiness, frustration, anger, and resolution states in real-time.
            </p>
          </div>

          <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-8 hover:bg-white/10 transition">
            <div className="w-14 h-14 bg-cyan-500/20 rounded-lg flex items-center justify-center mb-6">
              <TrendingUp className="w-7 h-7 text-cyan-400" />
            </div>
            <h3 className="text-2xl font-bold text-white mb-3">
              Performance Scoring
            </h3>
            <p className="text-gray-400">
              Comprehensive rubric evaluates helpfulness, respect, clarity, and
              policy adherence with explainable, evidence-based ratings.
            </p>
          </div>

          <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-8 hover:bg-white/10 transition">
            <div className="w-14 h-14 bg-orange-500/20 rounded-lg flex items-center justify-center mb-6">
              <Users className="w-7 h-7 text-orange-400" />
            </div>
            <h3 className="text-2xl font-bold text-white mb-3">AI Coaching</h3>
            <p className="text-gray-400">
              Get personalized coaching tips with specific examples from actual
              calls. Evidence-based recommendations for continuous improvement.
            </p>
          </div>

          <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-8 hover:bg-white/10 transition">
            <div className="w-14 h-14 bg-pink-500/20 rounded-lg flex items-center justify-center mb-6">
              <Shield className="w-7 h-7 text-pink-400" />
            </div>
            <h3 className="text-2xl font-bold text-white mb-3">
              Smart Reports
            </h3>
            <p className="text-gray-400">
              Automated weekly and monthly reports with actionable insights,
              trends, and performance metrics for agents and managers.
            </p>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="container mx-auto px-6 py-20">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            How It Works
          </h2>
          <p className="text-xl text-gray-400">
            Simple, powerful workflow from recording to insights
          </p>
        </div>

        <div className="max-w-4xl mx-auto space-y-12">
          <div className="flex flex-col md:flex-row items-center gap-8">
            <div className="flex-shrink-0 w-16 h-16 bg-blue-500 rounded-full flex items-center justify-center text-2xl font-bold text-white">
              1
            </div>
            <div className="flex-1">
              <h3 className="text-2xl font-bold text-white mb-2">
                Record Calls
              </h3>
              <p className="text-gray-400">
                Upload call recordings or integrate with your phone system.
                Support for single or dual-channel audio with automatic metadata
                capture.
              </p>
            </div>
          </div>

          <div className="flex flex-col md:flex-row items-center gap-8">
            <div className="flex-shrink-0 w-16 h-16 bg-purple-500 rounded-full flex items-center justify-center text-2xl font-bold text-white">
              2
            </div>
            <div className="flex-1">
              <h3 className="text-2xl font-bold text-white mb-2">
                AI Analysis
              </h3>
              <p className="text-gray-400">
                Our AI pipeline transcribes, identifies speakers, detects
                emotions, analyzes behavior, and scores performance against your
                quality rubric.
              </p>
            </div>
          </div>

          <div className="flex flex-col md:flex-row items-center gap-8">
            <div className="flex-shrink-0 w-16 h-16 bg-green-500 rounded-full flex items-center justify-center text-2xl font-bold text-white">
              3
            </div>
            <div className="flex-1">
              <h3 className="text-2xl font-bold text-white mb-2">
                Get Insights
              </h3>
              <p className="text-gray-400">
                View detailed call analysis with transcripts, emotion timelines,
                coaching tips, and performance scores. Export reports or
                integrate with your CRM.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section id="benefits" className="container mx-auto px-6 py-20">
        <div className="max-w-6xl mx-auto">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
                Measurable Results for Your Team
              </h2>
              <div className="space-y-6">
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 w-8 h-8 bg-green-500/20 rounded-lg flex items-center justify-center">
                    <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-white mb-2">
                      Reduce Handle Time
                    </h3>
                    <p className="text-gray-400">
                      Average 23% reduction in call duration through targeted
                      coaching and best practice identification.
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center">
                    <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-white mb-2">
                      Improve CSAT
                    </h3>
                    <p className="text-gray-400">
                      Customer satisfaction scores increase by 18% on average
                      with evidence-based performance feedback.
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 w-8 h-8 bg-purple-500/20 rounded-lg flex items-center justify-center">
                    <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-white mb-2">
                      Scale QA Efficiently
                    </h3>
                    <p className="text-gray-400">
                      Analyze 100% of calls automatically instead of random
                      sampling. Catch issues before they escalate.
                    </p>
                  </div>
                </div>
              </div>
            </div>
            <div className="bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-2xl p-8 border border-white/10 backdrop-blur-sm">
              <div className="space-y-6">
                <div className="bg-slate-900/80 rounded-lg p-6">
                  <div className="text-gray-400 text-sm mb-2">
                    Average Score Improvement
                  </div>
                  <div className="text-5xl font-bold text-white mb-4">+34%</div>
                  <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-blue-500 to-green-500 w-3/4"></div>
                  </div>
                </div>
                <div className="bg-slate-900/80 rounded-lg p-6">
                  <div className="text-gray-400 text-sm mb-2">
                    Calls Analyzed Monthly
                  </div>
                  <div className="text-5xl font-bold text-white">10K+</div>
                </div>
                <div className="bg-slate-900/80 rounded-lg p-6">
                  <div className="text-gray-400 text-sm mb-2">
                    Agent Satisfaction
                  </div>
                  <div className="text-5xl font-bold text-white">4.8/5</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="container mx-auto px-6 py-20">
        <div className="bg-gradient-to-r from-blue-600 to-cyan-600 rounded-3xl p-12 md:p-16 text-center">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
            Ready to Transform Your Support Team?
          </h2>
          <p className="text-xl text-blue-100 mb-10 max-w-2xl mx-auto">
            Join hundreds of companies using AI-powered insights to deliver
            exceptional customer experiences.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/signup"
              className="px-8 py-4 bg-white text-blue-600 hover:bg-gray-100 rounded-lg transition font-semibold text-lg"
            >
              Start Your Free Trial
            </Link>
            <a
              href="#contact"
              className="px-8 py-4 bg-transparent border-2 border-white text-white hover:bg-white/10 rounded-lg transition font-semibold text-lg"
            >
              Schedule a Demo
            </a>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="container mx-auto px-6 py-12 border-t border-white/10">
        <div className="grid md:grid-cols-4 gap-8 mb-8">
          <div>
            <div className="flex items-center space-x-2 mb-4">
              <div className="w-8 h-8 bg-blue-500 rounded-lg flex items-center justify-center">
                <Mic className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-bold text-white">AgentSights</span>
            </div>
            <p className="text-gray-400 text-sm">
              AI-powered customer support analytics and coaching platform.
            </p>
          </div>
          <div>
            <h4 className="text-white font-semibold mb-4">Product</h4>
            <ul className="space-y-2 text-gray-400 text-sm">
              <li>
                <a href="#" className="hover:text-white transition">
                  Features
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-white transition">
                  Pricing
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-white transition">
                  Integrations
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-white transition">
                  API Docs
                </a>
              </li>
            </ul>
          </div>
          <div>
            <h4 className="text-white font-semibold mb-4">Company</h4>
            <ul className="space-y-2 text-gray-400 text-sm">
              <li>
                <a href="#" className="hover:text-white transition">
                  About
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-white transition">
                  Blog
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-white transition">
                  Careers
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-white transition">
                  Contact
                </a>
              </li>
            </ul>
          </div>
          <div>
            <h4 className="text-white font-semibold mb-4">Legal</h4>
            <ul className="space-y-2 text-gray-400 text-sm">
              <li>
                <a href="#" className="hover:text-white transition">
                  Privacy
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-white transition">
                  Terms
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-white transition">
                  Security
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-white transition">
                  Compliance
                </a>
              </li>
            </ul>
          </div>
        </div>
        <div className="border-t border-white/10 pt-8 text-center text-gray-400 text-sm">
          <p>&copy; 2025 AgentSights. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}
