import json
import re
from typing import List, Dict, Set
from collections import defaultdict

from data_models.data_models import Turn


class TopicAnalyzer:
    
    def __init__(self, custom_keywords: Dict[str, Dict] = None):
        self.topic_patterns = {
            "delivery_shipping": {
                "keywords": ["deliver", "delivery", "ship", "shipping", "shipped", "arrive", "arrived", 
                           "eta", "tracking", "track", "transit", "courier", "transport", "freight",
                           "pickup", "drop off", "on the way", "en route"],
                "phrases": ["when will it arrive", "where is my", "hasn't arrived", 
                          "delivery date", "shipping address", "tracking number", "on its way"],
                "weight": 1.0
            },
            "returns_exchanges": {
                "keywords": ["return", "exchange", "refund", "send back", "wrong", "incorrect",
                           "damaged", "defective", "replace", "replacement", "swap", "change"],
                "phrases": ["want to return", "how do i return", "return policy", "exchange for",
                          "get a refund", "send it back", "not what i ordered"],
                "weight": 1.0
            },
            "payment_billing": {
                "keywords": ["payment", "charge", "charged", "credit card", "debit", "invoice",
                           "bill", "billing", "cost", "price", "paid", "pay", "transaction", "fee",
                           "money", "amount", "total", "balance", "autopay"],
                "phrases": ["payment method", "how much", "total cost", "payment failed",
                          "charge me", "billing statement", "payment plan"],
                "weight": 1.0
            },
            "product_service_inquiry": {
                "keywords": ["product", "item", "service", "option", "availability", "available",
                           "stock", "in stock", "specification", "feature", "details", "information",
                           "offer", "provide", "include", "come with"],
                "phrases": ["do you have", "is it available", "tell me about", "what is",
                          "how does it work", "what are the options", "product information"],
                "weight": 0.9
            },
            "order_status": {
                "keywords": ["order", "order number", "status", "confirmation", "placed", "processing",
                           "pending", "complete", "ready", "approved"],
                "phrases": ["order status", "where is my order", "order confirmation", 
                          "check my order", "when was it", "has it been"],
                "weight": 1.0
            },
            "account_access": {
                "keywords": ["account", "login", "password", "username", "profile", "register",
                           "sign in", "log in", "access", "portal", "dashboard"],
                "phrases": ["can't login", "forgot my password", "account locked", 
                          "create an account", "account information", "sign up"],
                "weight": 1.0
            },
            "technical_issues": {
                "keywords": ["not working", "broken", "error", "issue", "problem", "malfunction",
                           "fix", "repair", "troubleshoot", "glitch", "bug", "fail", "failed"],
                "phrases": ["doesn't work", "stopped working", "how do i fix", "technical issue",
                          "error message", "not functioning", "isn't responding"],
                "weight": 1.0
            },
            "cancellation": {
                "keywords": ["cancel", "cancelled", "cancellation", "stop", "terminate", "end",
                           "discontinue", "suspend"],
                "phrases": ["want to cancel", "cancel my", "stop the", "don't want", "terminate service"],
                "weight": 1.0
            },
            "complaint": {
                "keywords": ["complaint", "disappointed", "upset", "angry", "terrible", "awful",
                           "horrible", "unacceptable", "frustrated", "unhappy", "dissatisfied"],
                "phrases": ["not happy", "very disappointed", "speak to manager", "file a complaint",
                          "this is unacceptable", "waste of time"],
                "weight": 1.1
            },
            "pricing_cost": {
                "keywords": ["discount", "coupon", "promo", "promotion", "sale", "cheaper", "expensive",
                           "price", "rate", "quote", "estimate", "deal", "special"],
                "phrases": ["how much does it cost", "do you have a discount", "on sale",
                          "lower price", "best price", "price match"],
                "weight": 0.9
            },
            "appointment_scheduling": {
                "keywords": ["appointment", "schedule", "reschedule", "book", "booking", "reservation",
                           "slot", "time", "date", "available", "calendar"],
                "phrases": ["book an appointment", "schedule a", "change my appointment",
                          "when can i", "available times", "make a reservation"],
                "weight": 0.9
            },
            "contract_agreement": {
                "keywords": ["contract", "agreement", "terms", "lease", "subscription", "plan",
                           "commitment", "duration", "period"],
                "phrases": ["sign up for", "contract terms", "subscription plan", "agreement details",
                          "cancel my subscription"],
                "weight": 0.8
            },
            "location_address": {
                "keywords": ["location", "address", "where", "directions", "nearby", "close", "branch",
                           "office", "store", "facility", "site"],
                "phrases": ["where is", "closest to", "near me", "store location", "your address",
                          "how do i get to"],
                "weight": 0.7
            },
            "hours_availability": {
                "keywords": ["hours", "open", "close", "closed", "available", "business hours",
                           "operating", "schedule", "time"],
                "phrases": ["what time", "when are you open", "business hours", "operating hours",
                          "open today", "available when"],
                "weight": 0.7
            },
            "policy_terms": {
                "keywords": ["policy", "terms", "conditions", "rules", "guidelines", "procedure",
                           "regulation", "requirement"],
                "phrases": ["what is your policy", "company policy", "terms and conditions",
                          "according to policy", "policy states"],
                "weight": 0.8
            },
            "installation_setup": {
                "keywords": ["install", "installation", "setup", "activate", "activation", "configure",
                           "connect", "connection", "start", "initialize"],
                "phrases": ["how do i install", "set it up", "activate my", "connect to",
                          "installation process", "setup instructions"],
                "weight": 0.9
            },
            "upgrade_downgrade": {
                "keywords": ["upgrade", "downgrade", "change plan", "switch", "modify", "adjust",
                           "increase", "decrease", "level"],
                "phrases": ["upgrade my", "change my plan", "switch to", "downgrade to",
                          "better option", "different plan"],
                "weight": 0.8
            },
            "documentation_paperwork": {
                "keywords": ["document", "paperwork", "form", "application", "certificate", "proof",
                           "verification", "identification", "submit"],
                "phrases": ["need to send", "upload documents", "proof of", "fill out form",
                          "required documents", "paperwork needed"],
                "weight": 0.7
            },
            "transfer_redirect": {
                "keywords": ["transfer", "redirect", "forward", "escalate", "supervisor", "manager",
                           "department", "specialist"],
                "phrases": ["transfer me to", "speak to a manager", "escalate this", "another department",
                          "someone who can help"],
                "weight": 0.9
            },
            "followup_callback": {
                "keywords": ["follow up", "callback", "call back", "reach out", "contact", "get back",
                           "update", "status update"],
                "phrases": ["call me back", "follow up with", "get back to me", "contact me",
                          "let me know", "keep me updated"],
                "weight": 0.8
            }
        }
        
        if custom_keywords:
            for topic, patterns in custom_keywords.items():
                if topic in self.topic_patterns:
                    self.topic_patterns[topic]['keywords'].extend(patterns.get('keywords', []))
                    self.topic_patterns[topic]['phrases'].extend(patterns.get('phrases', []))
                else:
                    self.topic_patterns[topic] = patterns
    
    def analyze_call(self, turns: List[Dict]) -> Dict:
        if not turns:
            return self._empty_result()
        
        full_transcript = " ".join([t.get('text', '') for t in turns]).lower()
        customer_text = " ".join([t.get('text', '') for t in turns 
                                  if t.get('role', t.get('speaker', '')).lower().startswith('customer')]).lower()
        agent_text = " ".join([t.get('text', '') for t in turns 
                              if t.get('role', t.get('speaker', '')).lower().startswith('agent')]).lower()
        
        topic_scores = defaultdict(float)
        topic_evidence = defaultdict(list)
        
        for topic_name, patterns in self.topic_patterns.items():
            score, evidence = self._calculate_topic_score(
                full_transcript, 
                customer_text,
                patterns['keywords'], 
                patterns['phrases'],
                patterns['weight']
            )
            
            if score > 0:
                topic_scores[topic_name] = score
                topic_evidence[topic_name] = evidence
        
        topic_timeline = self._extract_topic_timeline(turns, topic_scores.keys())
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        
        primary_topic = sorted_topics[0][0] if sorted_topics else None
        primary_confidence = sorted_topics[0][1] if sorted_topics else 0.0
        
        secondary_topics = [
            {"topic": topic, "confidence": round(score, 2)}
            for topic, score in sorted_topics[1:4]
            if score >= 0.3
        ]
        
        customer_topics = self._identify_speaker_topics(customer_text)
        agent_topics = self._identify_speaker_topics(agent_text)
        
        return {
            "primary_topic": primary_topic,
            "primary_confidence": round(primary_confidence, 2),
            "secondary_topics": secondary_topics,
            "all_topics_detected": [
                {
                    "topic": topic,
                    "confidence": round(score, 2),
                    "evidence_count": len(topic_evidence[topic]),
                    "sample_evidence": topic_evidence[topic][:3]
                }
                for topic, score in sorted_topics
            ],
            "customer_topics": customer_topics,
            "agent_topics": agent_topics,
            "topic_timeline": topic_timeline,
            "total_topics_detected": len(topic_scores)
        }
    
    def _calculate_topic_score(self, text: str, customer_text: str, 
                               keywords: List[str], phrases: List[str], 
                               weight: float) -> tuple:
        score = 0.0
        evidence = []
        
        matched_keywords = set()
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, text):
                matched_keywords.add(keyword)
                score += 0.5
        
        matched_phrases = []
        for phrase in phrases:
            if phrase.lower() in text:
                matched_phrases.append(phrase)
                score += 1.0
        
        if matched_keywords:
            evidence.append(f"Keywords: {', '.join(list(matched_keywords)[:5])}")
        if matched_phrases:
            evidence.append(f"Phrases: {', '.join(matched_phrases[:3])}")
        
        if customer_text:
            customer_keyword_count = sum(1 for kw in keywords if kw.lower() in customer_text)
            if customer_keyword_count > 0:
                score *= 1.2
                evidence.append(f"Customer mentioned {customer_keyword_count} related keywords")
        
        score *= weight
        score = min(score, 10.0)
        
        return score, evidence
    
    def _extract_topic_timeline(self, turns: List[Dict], detected_topics: Set[str]) -> List[Dict]:
        timeline = []
        
        for turn_idx, turn in enumerate(turns):
            text = turn.get('text', '').lower()
            start_time = turn.get('start', turn_idx)
            speaker = turn.get('role', turn.get('speaker', 'Unknown'))
            
            for topic in detected_topics:
                patterns = self.topic_patterns[topic]
                mentioned = False
                matched_term = None
                
                for keyword in patterns['keywords']:
                    if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text):
                        mentioned = True
                        matched_term = keyword
                        break
                
                if not mentioned:
                    for phrase in patterns['phrases']:
                        if phrase.lower() in text:
                            mentioned = True
                            matched_term = phrase
                            break
                
                if mentioned:
                    timeline.append({
                        "timestamp": start_time,
                        "turn_index": turn_idx,
                        "topic": topic,
                        "speaker": speaker,
                        "matched_term": matched_term,
                        "context": text[:100]
                    })
        
        timeline.sort(key=lambda x: x['timestamp'])
        return timeline
    
    def _identify_speaker_topics(self, speaker_text: str) -> List[str]:
        topics = []
        
        for topic_name, patterns in self.topic_patterns.items():
            keyword_matches = sum(1 for kw in patterns['keywords'] 
                                 if re.search(r'\b' + re.escape(kw.lower()) + r'\b', speaker_text))
            phrase_matches = sum(1 for ph in patterns['phrases'] if ph.lower() in speaker_text)
            
            if keyword_matches >= 2 or phrase_matches >= 1:
                topics.append(topic_name)
        
        return topics
    
    def _empty_result(self) -> Dict:
        return {
            "primary_topic": None,
            "primary_confidence": 0.0,
            "secondary_topics": [],
            "all_topics_detected": [],
            "customer_topics": [],
            "agent_topics": [],
            "topic_timeline": [],
            "total_topics_detected": 0
        }


_topic_analyzer_instance = None


def get_topic_analyzer() -> "TopicAnalyzer":
    global _topic_analyzer_instance
    if _topic_analyzer_instance is None:
        _topic_analyzer_instance = TopicAnalyzer()
    return _topic_analyzer_instance


def analyze_topics(transcript: dict, custom_keywords: Dict[str, Dict] = None) -> Dict:
    analyzer = TopicAnalyzer(custom_keywords=custom_keywords) if custom_keywords else get_topic_analyzer()
    
    if 'utterances' in transcript:
        turns = transcript['utterances']
    elif 'segments' in transcript:
        turns = [{"text": seg.get('text', ''), "speaker": seg.get('speaker', 'Unknown')} 
                 for seg in transcript['segments']]
    else:
        turns = []
    
    return analyzer.analyze_call(turns)
