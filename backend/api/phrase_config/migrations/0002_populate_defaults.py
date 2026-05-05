"""
Data migration: seed all default phrase lists into phrase_config_phraselist.

All values are copied verbatim from the AI module constants as of this migration.
Uses ignore_conflicts=True so it is safe to re-run (idempotent).
"""

from django.db import migrations


def populate_phrase_lists(apps, schema_editor):
    PhraseList = apps.get_model("phrase_config", "PhraseList")

    entries = [
        # ── whisperer.py ──────────────────────────────────────────────────────
        {
            "name": "agent_greeting_phrases",
            "list_type": "phrase_list",
            "description": "Canonical agent opening phrases used to identify the agent speaker by greeting (first 8 utterances).",
            "data": [
                "thank you for calling",
                "thanks for calling",
                "thank you for contacting",
                "thanks for contacting",
                "thank you for reaching",
                "thanks for reaching",
                "how may i help you",
                "how can i help you",
                "how may i assist you",
                "how can i assist you",
                "how can i be of assistance",
                "how may i be of assistance",
                "you've reached",
                "you have reached",
                "welcome to",
                "speaking, how",
                "my name is",
                "good morning",
                "good afternoon",
                "good evening",
                "i'll be happy to help",
                "i'd be happy to help",
                "happy to help you today",
            ],
        },
        {
            "name": "announcement_keywords",
            "list_type": "phrase_list",
            "description": "Words that indicate a recording/monitoring announcement rather than agent speech. Used to skip non-agent utterances during agent detection.",
            "data": ["record", "recorded", "recording", "monitor", "monitored", "monitoring"],
        },
        {
            "name": "customer_phrases",
            "list_type": "phrase_list",
            "description": "Phrases strongly diagnostic of the customer speaking. Used by _score_speakers_by_phrases with weight -1 for agent scoring.",
            "data": [
                # Cancellation / disconnection
                "i'd like to cancel",
                "i want to cancel",
                "i need to cancel",
                "i'd like to disconnect",
                "i want to disconnect",
                "please cancel",
                "cancel my service",
                "cancel my account",
                "cancel my subscription",
                "close my account",
                "i want to close",
                "i'm calling to cancel",
                "we'd like to cancel",
                "we want to cancel",
                "we need to cancel",
                "we're looking to cancel",
                "we're calling to cancel",
                "we'd like to disconnect",
                "we want to disconnect",
                "we need to disconnect",
                "please cancel our service",
                "please cancel our account",
                "cancel our service",
                "cancel our account",
                "close our account",
                "we want to close",
                "just disconnect us",
                "just cancel our",
                "disconnect our service",
                # Refund / billing complaints
                "i'd like a refund",
                "i want a refund",
                "i need a refund",
                "i want my money back",
                "i need my money back",
                "you charged me",
                "you billed me",
                "you overcharged me",
                "i was overcharged",
                "i shouldn't be charged",
                "i didn't authorize",
                "i never agreed",
                "why was i charged",
                "why am i being charged",
                "this charge is wrong",
                "this bill is wrong",
                # Escalation
                "i want to speak to a manager",
                "i want to speak to your supervisor",
                "i'd like to speak to a manager",
                "speak to a manager",
                "get me a manager",
                "get me your supervisor",
                # Complaints / frustration
                "i have a problem",
                "i have an issue",
                "i'm having a problem",
                "i'm calling because",
                "this is wrong",
                "that's not right",
                "this is unacceptable",
                "this is ridiculous",
                "i've been waiting",
                "i've been on hold",
                "i've called multiple times",
                "no one has helped me",
                # Technical issues (customer perspective)
                "my service is not working",
                "my internet is not working",
                "it's not working",
                "i can't connect",
                "i've already tried",
                "i already restarted",
                "i already rebooted",
            ],
        },
        {
            "name": "agent_phrases",
            "list_type": "phrase_list",
            "description": "Phrases strongly diagnostic of the agent speaking. Used by _score_speakers_by_phrases with weight +2 for agent scoring.",
            "data": [
                # Standard greetings / closings
                "thank you for calling",
                "thanks for calling",
                "how can i help you",
                "how may i help you",
                "i can help you with that",
                "i'll be happy to help",
                "i'd be happy to help",
                "is there anything else i can",
                "have a great day",
                "have a wonderful day",
                "thank you for your time",
                "thank you for your patience",
                "thank you for waiting",
                "please don't hesitate to call",
                # Account lookup / procedural
                "let me pull up your account",
                "let me look up your account",
                "let me look into that for you",
                "let me look that up",
                "let me go ahead and",
                "i'm going to go ahead",
                "can i place you on a brief hold",
                "can i put you on hold",
                "can i transfer you",
                "i can transfer you",
                "i'll transfer you to",
                "for verification purposes",
                "for security purposes",
                "can i verify your",
                # System / account observations
                "i can see on your account",
                "looking at your account",
                "according to our records",
                "our system shows",
                "i see here that",
                "your account shows",
                "i'm looking at your account",
                "let me make a note",
                "i'll update your account",
                # Technical support actions
                "i'll create a ticket",
                "i'll escalate this",
                "a technician will",
                "we'll send a technician",
                "your ticket number is",
                "your reference number is",
                "let me reset your",
                "i can reset your",
                # Billing actions
                "i can apply a credit",
                "let me apply a credit",
                "i'll process a refund",
                "i can reverse that charge",
                "i can waive that fee",
                "i can adjust your bill",
                "i'll credit your account",
                # Retention / offers
                "before i process that",
                "before i do that",
                "before i go ahead",
                "before i can process",
                "before i can cancel",
                "before i cancel",
                "before i disconnect",
                "before we proceed",
                "let me see what i can offer",
                "let me see what offers",
                "let me see what i can do for you",
                "let me see what i can do",
                "see what i can do",
                "what i can offer",
                "what we can offer",
                "let me pull up some offers",
                "let me check what deals",
                "i have some great offers",
                "we have a great offer",
                "i can offer you",
                "what if i offered",
                "i'd hate to lose you",
                "i'd hate to see you go",
                "hate to lose you",
                "hate to see you go",
                "we value you as a customer",
                "instead of cancelling",
                "instead of disconnecting",
                "have you considered keeping",
                "my job is to",
                "about keeping your service",
                "about keeping your",
                "keeping your service",
                "retain your service",
                "about retaining your",
                "reason for cancelling",
                "reason for disconnecting",
                "looking to cancel today",
                "looking to disconnect",
                "can i ask why",
                "may i ask why",
                "understand you'd like to cancel",
                "understand you want to cancel",
                "understand you'd like to disconnect",
                # Understanding / empathy (agent framing)
                "i understand your frustration",
                "i understand your concern",
                "i apologize for the inconvenience",
                "i'm sorry to hear that",
                "i'm sorry about that",
                "what seems to be the issue",
                "can you describe the issue",
            ],
        },
        {
            "name": "agent_exclusive_phrases",
            "list_type": "phrase_list",
            "description": "Phrases so role-specific that their presence in a Customer turn indicates a PyAnnote mis-assignment. Used by _relabel_turns_by_content. Covers all major call types.",
            "data": [
                # Opening / greeting
                "thank you for calling",
                "thanks for calling",
                "thank you for contacting",
                "you've reached",
                "you have reached",
                "how may i help you today",
                "how can i help you today",
                "how may i assist you today",
                "how can i assist you today",
                "i'll be happy to help",
                "i'd be happy to help",
                "i'm happy to assist",
                "certainly, i can help",
                "of course, i can help",
                "i'd be glad to help",
                # Account verification
                "can i get your account number",
                "can i have your account number",
                "for verification purposes",
                "for security purposes",
                "to verify your identity",
                "to verify your account",
                "let me pull up your account",
                "let me look up your account",
                "i'll look up your account",
                "let me bring up your account",
                "let me find your account",
                # Account observations
                "i can see on your account",
                "i can see here that",
                "looking at your account",
                "according to our records",
                "according to your account",
                "our system shows",
                "i see here that",
                "on your account it shows",
                "your account shows",
                "i'm looking at your account",
                "i can see the issue on our end",
                "i can see your account",
                "i have your account",
                "the account shows",
                "i can see that on your account",
                # Hold / transfer
                "let me place you on hold",
                "can i put you on hold",
                "please hold for a moment",
                "one moment please",
                "please hold for",
                "i'm going to place you on hold",
                "i'm going to transfer you",
                "let me transfer you",
                "i'll transfer you to",
                "i'm transferring you",
                "let me connect you with",
                "i'll connect you to",
                "let me put you through to",
                # Notes / account updates
                "i'm going to note",
                "i've noted on your account",
                "let me make a note",
                "i'll make a note",
                "i'll update your account",
                "i've updated your account",
                "i'll update that on your account",
                "i'm updating your record",
                "i'm updating your account",
                "that has been updated",
                "i've processed that",
                "i'm going to document",
                "i'll add that to your account",
                "i've made note of that",
                # Processing requests
                "i'll process that for you",
                "let me process that",
                "i'm processing your request",
                "let me go ahead and process",
                "i can process that",
                "before i process that",
                "before i can process",
                "before i go ahead and cancel",
                "before i go ahead with that",
                "before i can cancel",
                "before i cancel",
                "before i disconnect",
                "before i do that",
                "before i go ahead",
                "before we proceed",
                # Technical support (agent actions)
                "let me run a diagnostic",
                "let me run a test",
                "i can reset your",
                "let me reset your",
                "i'll reset your password",
                "i can reset your password",
                "i'll create a ticket",
                "i'll open a ticket",
                "i'm going to create a ticket",
                "i'll escalate this",
                "i'm escalating this",
                "i'm going to escalate",
                "i'll escalate that",
                "a technician will",
                "we'll send a technician",
                "i'll have a technician",
                "an engineer will",
                "your ticket number is",
                "your case number is",
                "your reference number is",
                "i'll have that fixed",
                "i can see an outage",
                "there is an outage",
                "we're aware of the issue",
                "our team is working on it",
                "i'll push an update",
                "let me push a refresh",
                "i'm going to refresh your service",
                "let me refresh your line",
                "let me reset the connection",
                # Billing actions (agent initiates)
                "i can apply a credit",
                "let me apply a credit",
                "i'll process a refund",
                "let me process a refund",
                "i can reverse that charge",
                "i can waive that fee",
                "let me waive the",
                "i can remove that charge",
                "i'll remove that charge",
                "that charge was for",
                "i can adjust your bill",
                "i'm going to credit your account",
                "i'll credit your account",
                "i can issue a credit",
                "the charge on your account",
                "i can refund that",
                "i'll issue a refund",
                "i can process a refund",
                "i'll reverse that",
                "i can waive the late fee",
                "i'll remove the fee",
                # Offers / retention / upsell
                "i can offer you",
                "what if i offered",
                "what if i could offer",
                "let me see what i can offer",
                "let me see what i can do for you",
                "let me see what deals",
                "let me check what deals",
                "let me see what offers",
                "let me pull up some offers",
                "we have a promotion",
                "we have a special offer",
                "i have some great deals",
                "i have some great offers",
                "we have a great offer",
                "i can give you a discount",
                "give you a discount",
                "offer you a discount",
                "offer you a better rate",
                "i can reduce your",
                "i can reduce your bill",
                "i can reduce your rate",
                "i can bring down",
                "bring down your bill",
                "what if i could save you",
                "what if i could bring",
                "what if i could bring down",
                "promotional rate",
                "promotional offer",
                "loyalty discount",
                "special offer for you",
                "let me see what i can do",
                "see what i can do",
                "what i can offer",
                "what we can offer",
                "instead of cancelling",
                "instead of disconnecting",
                "have you considered keeping",
                "keeping your service",
                "about keeping your service",
                "about keeping your account",
                "retain your service",
                "about retaining your",
                "to keep you as a customer",
                "i'd like to keep you",
                "i'd hate to lose you",
                "hate to lose you",
                "hate to see you go",
                "would hate to lose",
                "we value you as a customer",
                "my job is to",
                # Understanding / retention framing
                "i understand you'd like to cancel",
                "i understand you want to cancel",
                "understand you're looking to cancel",
                "i understand you'd like to disconnect",
                "reason you're looking to cancel",
                "reason for cancelling",
                "reason for disconnecting",
                "looking to cancel today",
                "looking to disconnect",
                "can i ask the reason",
                "may i ask the reason",
                "can i ask why",
                "may i ask why",
                # Empathy / professional courtesy
                "i understand your frustration",
                "i understand your concern",
                "i appreciate your patience",
                "thank you for your patience",
                "thank you for waiting",
                "i apologize for the inconvenience",
                "i apologize for any inconvenience",
                "i'm sorry to hear that",
                "i'm sorry about that",
                "i sincerely apologize",
                # Questions only an agent asks
                "what seems to be the issue",
                "what can i help you with",
                "what is the issue today",
                "can you describe the issue",
                "can you describe the problem",
                "is there anything i can do",
                "anything i can do to keep",
                "happy to help you with that",
                # Closing
                "is there anything else i can",
                "is there anything else i can help",
                "anything else i can help you with",
                "have a great day",
                "have a wonderful day",
                "thank you for your time",
                "thanks for your time",
                "please don't hesitate to call back",
                "don't hesitate to call us",
                "have a good one",
            ],
        },
        {
            "name": "customer_exclusive_phrases",
            "list_type": "phrase_list",
            "description": "Phrases so role-specific that their presence in an Agent turn indicates a PyAnnote mis-assignment. Used by _relabel_turns_by_content.",
            "data": [
                # Cancellation / disconnection requests
                "i'd like to cancel",
                "i want to cancel",
                "i need to cancel",
                "please cancel",
                "cancel my service",
                "cancel my account",
                "cancel my subscription",
                "close my account",
                "i want to close my account",
                "we'd like to cancel",
                "we want to cancel",
                "we need to cancel",
                "we're looking to cancel",
                "we're calling to cancel",
                "we'd like to disconnect",
                "we want to disconnect",
                "we need to disconnect",
                "just disconnect us",
                "just cancel our",
                "please cancel our service",
                "cancel our service",
                "cancel our account",
                "close our account",
                "disconnect our service",
                "i'm done with this service",
                "done with the service",
                "no longer want the service",
                "no longer need the service",
                "stop my service",
                "end my service",
                "terminate my service",
                "terminate my account",
                "i'm calling to cancel",
                "just want to cancel",
                "please just cancel",
                "just cancel it",
                "just process the cancellation",
                "please process the cancellation",
                "please disconnect my service",
                "please disconnect my account",
                # Refund / return requests
                "i'd like a refund",
                "i want a refund",
                "i need a refund",
                "please give me a refund",
                "i want my money back",
                "i need my money back",
                "i'd like to return",
                "i want to return",
                "i need to return",
                "please process my return",
                "i want to send it back",
                # Billing complaints (customer's direct experience)
                "you charged me",
                "you overcharged me",
                "i was overcharged",
                "i shouldn't be charged",
                "i didn't authorize",
                "i never agreed to",
                "this charge is wrong",
                "this bill is wrong",
                "why was i charged",
                "why am i being charged",
                "why is my bill",
                "i'm being overcharged",
                "i'm paying too much",
                "my bill is too high",
                "you billed me",
                "that charge is incorrect",
                "i never authorized",
                "i did not authorize",
                # Rejecting retention offers
                "we're not interested",
                "i'm not interested in any offers",
                "not interested in offers",
                "no offers please",
                "don't want any offers",
                "i don't want any deals",
                "we don't want any deals",
                "not interested in any deals",
                "please just process it",
                "just process it please",
                "i've made my decision",
                "we've made our decision",
                "my decision is final",
                "our decision is final",
                "please stop and",
                "stop trying to keep",
                "i don't want to discuss",
                "please don't offer me",
                "i'm not interested in keeping",
                "we're not interested in keeping",
                # Escalation requests
                "i want to speak to a manager",
                "i want to speak to your supervisor",
                "i'd like to speak to a manager",
                "i need to speak to a manager",
                "let me speak to a manager",
                "speak to your manager",
                "speak to your supervisor",
                "get me your supervisor",
                "get me a manager",
                "i want to speak to someone else",
                "i need to speak to someone else",
                "escalate this to a manager",
                "escalate my complaint",
                # Frustration / complaint expressions
                "this is unacceptable",
                "that's not acceptable",
                "this is ridiculous",
                "this is a joke",
                "i'm very frustrated",
                "i'm extremely frustrated",
                "i'm so frustrated",
                "i'm so angry",
                "i'm fed up",
                "i'm sick of this",
                "i've had enough",
                "i've been waiting",
                "i've been on hold for",
                "i've called multiple times",
                "i've already called",
                "i already spoke to",
                "no one has helped me",
                "this keeps happening",
                "i've been dealing with this",
                "this is terrible service",
                "i'm very disappointed",
                "i'm extremely disappointed",
                "will not give you that",
                # Service / product problems
                "my service is not working",
                "my internet is not working",
                "my device is not working",
                "it's not working",
                "it stopped working",
                "it's been down since",
                "i can't connect",
                "i'm not getting service",
                "i've been without service",
                "my service has been down",
                "the problem started",
                "i already tried that",
                "i've already tried",
                "i already restarted",
                "i already rebooted",
                "nothing is working",
                "i've done everything",
                "i tried rebooting",
                "i tried restarting",
                "i followed the instructions",
                # Tenure / loyalty expressions
                "i've been with you for",
                "i've been a customer for",
                "i've been your customer",
                "i'm a long-time customer",
                "after all these years",
                "i've been loyal",
                "i've been paying for",
                # Decision finality / direct demands
                "please just do it",
                "can you just",
                "why can't you just",
                "just do what i'm asking",
                "you're not listening",
                "you're not hearing me",
                "i've already said",
                "i've said it multiple times",
                "i don't understand why",
                "i shouldn't have to",
                "this should be simple",
            ],
        },

        # ── local_summary.py ──────────────────────────────────────────────────
        {
            "name": "request_words",
            "list_type": "phrase_list",
            "description": "Words/phrases counted in _apply_outcome_correction to detect unresolved customer requests. 3+ hits = refusal scenario; 5+ = helpfulness capped at 2; 10+ = respect/adherence capped.",
            "data": [
                "cancel",
                "cancellation",
                "cancelled",
                "disconnect",
                "disconnecting",
                "close my account",
                "close the account",
                "speak to a manager",
                "speak to your supervisor",
                "speak to a supervisor",
                "transfer me",
                "refund",
                "billing error",
                "overcharged",
                "charged me wrong",
            ],
        },
        {
            "name": "refusal_signals",
            "list_type": "phrase_list",
            "description": "Agent phrases that indicate blocking/retention behaviour. Any one match in _apply_outcome_correction confirms the agent was resisting the customer's request.",
            "data": [
                # Inability / refusal
                "i'm not able to",
                "i am not able to",
                "i can't do that",
                "i cannot do that",
                "that's not something i can",
                "that's not something we can",
                "that's not possible",
                "i don't have the ability",
                "my hands are tied",
                # Deflection / stalling
                "i understand but",
                "i hear you but",
                "i know but",
                "before i do that",
                "before i go ahead",
                "before i process",
                "before we proceed",
                # Retention offers
                "let me see what i can do for you",
                "let me see what i can do",
                "let me see what offers",
                "let me see what i can offer",
                "let me pull up some offers",
                "let me check what deals",
                "i have some great offers",
                "we have a great offer",
                "i can offer you",
                "what if i offered",
                "i'd hate to lose you",
                "i'd hate to see you go",
                "we value you as a customer",
                # Ignoring the request
                "instead of cancelling",
                "instead of disconnecting",
                "have you considered",
                "what if instead",
            ],
        },

        # ── emotion_analyzer.py ───────────────────────────────────────────────
        {
            "name": "apology_phrases",
            "list_type": "phrase_list",
            "description": "Agent apology phrases used by _detect_apology_bert to set turn.contains_apology=True.",
            "data": [
                "i'm sorry",
                "i am sorry",
                "so sorry",
                "very sorry",
                "i apologize",
                "my apologies",
                "apologies for",
                "sorry for",
                "sorry about",
                "i regret",
            ],
        },
        {
            "name": "empathy_phrases",
            "list_type": "phrase_list",
            "description": "Agent empathy phrases used by _detect_empathy_bert to set turn.contains_empathy=True.",
            "data": [
                "i understand",
                "i can understand",
                "i completely understand",
                "i realize",
                "i can see",
                "i appreciate",
                "thank you for",
                "thanks for",
                "that must be",
                "that sounds",
                "i hear you",
                "you're right",
                "that's frustrating",
                "that's disappointing",
                "i can imagine",
            ],
        },

        # ── Emotion_analyzation/summary.py ────────────────────────────────────
        {
            "name": "conclusive_phrases",
            "list_type": "phrase_list",
            "description": "Agent closing phrases that indicate the call was properly concluded. Used by _determine_resolution to set agent_closed_call=True.",
            "data": [
                "anything else",
                "have a great",
                "have a good",
                "goodbye",
                "take care",
                "you're all set",
                "all set",
                "you're welcome",
                "you are welcome",
                "glad i could",
                "happy to help",
                "is there anything",
                "good day",
            ],
        },

        # ── orchestrator.py ───────────────────────────────────────────────────
        {
            "name": "tone_to_satisfaction",
            "list_type": "mapping",
            "description": "Maps local model customer_tone labels to customer_satisfaction values in the emotion summary.",
            "data": {
                "positive": "very_satisfied",
                "satisfied": "satisfied",
                "neutral": "neutral",
                "frustrated": "dissatisfied",
                "negative": "dissatisfied",
                "angry": "very_dissatisfied",
            },
        },
        {
            "name": "tone_to_call_tone",
            "list_type": "mapping",
            "description": "Maps local model customer_tone labels to the call_tone field in the emotion summary.",
            "data": {
                "positive": "positive",
                "satisfied": "positive",
                "neutral": "neutral",
                "frustrated": "negative",
                "negative": "negative",
                "angry": "negative",
            },
        },
        {
            "name": "tone_to_resolution",
            "list_type": "mapping",
            "description": "Maps local model customer_tone labels to resolution_status. Used to override the rule-based resolution when local model tone is strongly negative.",
            "data": {
                "positive": "resolved",
                "satisfied": "resolved",
                "neutral": "pending",
                "frustrated": "unresolved",
                "negative": "unresolved",
                "angry": "unresolved",
            },
        },
        {
            "name": "agent_tone_to_empathy",
            "list_type": "mapping",
            "description": "Maps local model agent_tone labels to agent_empathy_score values (0.0–1.0). Blended with rule-based score (max of both).",
            "data": {
                "empathetic": 0.85,
                "apologetic": 0.75,
                "helpful": 0.6,
                "professional": 0.5,
                "positive": 0.55,
                "neutral": 0.3,
                "dismissive": 0.1,
            },
        },

        # ── topic_analyzer.py ─────────────────────────────────────────────────
        {
            "name": "topic_patterns",
            "list_type": "topic_group",
            "description": "Default topic detection patterns for TopicAnalyzer. Each key is a topic name; value has keywords, phrases, and weight. Company custom_keywords are merged on top at runtime.",
            "data": {
                "delivery_shipping": {
                    "keywords": [
                        "deliver", "delivery", "ship", "shipping", "shipped", "arrive", "arrived",
                        "eta", "tracking", "track", "transit", "courier", "transport", "freight",
                        "pickup", "drop off", "on the way", "en route"
                    ],
                    "phrases": [
                        "when will it arrive", "where is my", "hasn't arrived",
                        "delivery date", "shipping address", "tracking number", "on its way"
                    ],
                    "weight": 1.0
                },
                "returns_exchanges": {
                    "keywords": [
                        "return", "exchange", "refund", "send back", "wrong", "incorrect",
                        "damaged", "defective", "replace", "replacement", "swap", "change"
                    ],
                    "phrases": [
                        "want to return", "how do i return", "return policy", "exchange for",
                        "get a refund", "send it back", "not what i ordered"
                    ],
                    "weight": 1.0
                },
                "payment_billing": {
                    "keywords": [
                        "payment", "charge", "charged", "credit card", "debit", "invoice",
                        "bill", "billing", "cost", "price", "paid", "pay", "transaction", "fee",
                        "money", "amount", "total", "balance", "autopay"
                    ],
                    "phrases": [
                        "payment method", "how much", "total cost", "payment failed",
                        "charge me", "billing statement", "payment plan"
                    ],
                    "weight": 1.0
                },
                "product_service_inquiry": {
                    "keywords": [
                        "product", "item", "service", "option", "availability", "available",
                        "stock", "in stock", "specification", "feature", "details", "information",
                        "offer", "provide", "include", "come with"
                    ],
                    "phrases": [
                        "do you have", "is it available", "tell me about", "what is",
                        "how does it work", "what are the options", "product information"
                    ],
                    "weight": 0.9
                },
                "order_status": {
                    "keywords": [
                        "order", "order number", "status", "confirmation", "placed", "processing",
                        "pending", "complete", "ready", "approved"
                    ],
                    "phrases": [
                        "order status", "where is my order", "order confirmation",
                        "check my order", "when was it", "has it been"
                    ],
                    "weight": 1.0
                },
                "account_access": {
                    "keywords": [
                        "account", "login", "password", "username", "profile", "register",
                        "sign in", "log in", "access", "portal", "dashboard"
                    ],
                    "phrases": [
                        "can't login", "forgot my password", "account locked",
                        "create an account", "account information", "sign up"
                    ],
                    "weight": 1.0
                },
                "technical_issues": {
                    "keywords": [
                        "not working", "broken", "error", "issue", "problem", "malfunction",
                        "fix", "repair", "troubleshoot", "glitch", "bug", "fail", "failed"
                    ],
                    "phrases": [
                        "doesn't work", "stopped working", "how do i fix", "technical issue",
                        "error message", "not functioning", "isn't responding"
                    ],
                    "weight": 1.0
                },
                "cancellation": {
                    "keywords": [
                        "cancel", "cancelled", "cancellation", "stop", "terminate", "end",
                        "discontinue", "suspend"
                    ],
                    "phrases": [
                        "want to cancel", "cancel my", "stop the", "don't want", "terminate service"
                    ],
                    "weight": 1.0
                },
                "complaint": {
                    "keywords": [
                        "complaint", "disappointed", "upset", "angry", "terrible", "awful",
                        "horrible", "unacceptable", "frustrated", "unhappy", "dissatisfied"
                    ],
                    "phrases": [
                        "not happy", "very disappointed", "speak to manager", "file a complaint",
                        "this is unacceptable", "waste of time"
                    ],
                    "weight": 1.1
                },
                "pricing_cost": {
                    "keywords": [
                        "discount", "coupon", "promo", "promotion", "sale", "cheaper", "expensive",
                        "price", "rate", "quote", "estimate", "deal", "special"
                    ],
                    "phrases": [
                        "how much does it cost", "do you have a discount", "on sale",
                        "lower price", "best price", "price match"
                    ],
                    "weight": 0.9
                },
                "appointment_scheduling": {
                    "keywords": [
                        "appointment", "schedule", "reschedule", "book", "booking", "reservation",
                        "slot", "time", "date", "available", "calendar"
                    ],
                    "phrases": [
                        "book an appointment", "schedule a", "change my appointment",
                        "when can i", "available times", "make a reservation"
                    ],
                    "weight": 0.9
                },
                "contract_agreement": {
                    "keywords": [
                        "contract", "agreement", "terms", "lease", "subscription", "plan",
                        "commitment", "duration", "period"
                    ],
                    "phrases": [
                        "sign up for", "contract terms", "subscription plan", "agreement details",
                        "cancel my subscription"
                    ],
                    "weight": 0.8
                },
                "location_address": {
                    "keywords": [
                        "location", "address", "where", "directions", "nearby", "close", "branch",
                        "office", "store", "facility", "site"
                    ],
                    "phrases": [
                        "where is", "closest to", "near me", "store location", "your address",
                        "how do i get to"
                    ],
                    "weight": 0.7
                },
                "hours_availability": {
                    "keywords": [
                        "hours", "open", "close", "closed", "available", "business hours",
                        "operating", "schedule", "time"
                    ],
                    "phrases": [
                        "what time", "when are you open", "business hours", "operating hours",
                        "open today", "available when"
                    ],
                    "weight": 0.7
                },
                "policy_terms": {
                    "keywords": [
                        "policy", "terms", "conditions", "rules", "guidelines", "procedure",
                        "regulation", "requirement"
                    ],
                    "phrases": [
                        "what is your policy", "company policy", "terms and conditions",
                        "according to policy", "policy states"
                    ],
                    "weight": 0.8
                },
                "installation_setup": {
                    "keywords": [
                        "install", "installation", "setup", "activate", "activation", "configure",
                        "connect", "connection", "start", "initialize"
                    ],
                    "phrases": [
                        "how do i install", "set it up", "activate my", "connect to",
                        "installation process", "setup instructions"
                    ],
                    "weight": 0.9
                },
                "upgrade_downgrade": {
                    "keywords": [
                        "upgrade", "downgrade", "change plan", "switch", "modify", "adjust",
                        "increase", "decrease", "level"
                    ],
                    "phrases": [
                        "upgrade my", "change my plan", "switch to", "downgrade to",
                        "better option", "different plan"
                    ],
                    "weight": 0.8
                },
                "documentation_paperwork": {
                    "keywords": [
                        "document", "paperwork", "form", "application", "certificate", "proof",
                        "verification", "identification", "submit"
                    ],
                    "phrases": [
                        "need to send", "upload documents", "proof of", "fill out form",
                        "required documents", "paperwork needed"
                    ],
                    "weight": 0.7
                },
                "transfer_redirect": {
                    "keywords": [
                        "transfer", "redirect", "forward", "escalate", "supervisor", "manager",
                        "department", "specialist"
                    ],
                    "phrases": [
                        "transfer me to", "speak to a manager", "escalate this", "another department",
                        "someone who can help"
                    ],
                    "weight": 0.9
                },
                "followup_callback": {
                    "keywords": [
                        "follow up", "callback", "call back", "reach out", "contact", "get back",
                        "update", "status update"
                    ],
                    "phrases": [
                        "call me back", "follow up with", "get back to me", "contact me",
                        "let me know", "keep me updated"
                    ],
                    "weight": 0.8
                },
            },
        },
    ]

    PhraseList.objects.bulk_create(
        [PhraseList(**entry) for entry in entries],
        ignore_conflicts=True,
    )


def reverse_noop(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("phrase_config", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(populate_phrase_lists, reverse_code=reverse_noop),
    ]
