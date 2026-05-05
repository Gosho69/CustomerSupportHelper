import os, json, sys
import warnings
import soundfile as sf
import dotenv

warnings.filterwarnings("ignore", category=UserWarning, message=".*torchaudio.*deprecated.*")

import phrase_loader as _pl

# ---------------------------------------------------------------------------
# Agent-detection helpers
# ---------------------------------------------------------------------------

# Phrases that are strongly diagnostic of the *agent* speaking.  These are
# the standard call-centre greetings that appear at the start of a call.
_AGENT_GREETING_PHRASES_DEFAULT = [
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
    "speaking, how",        # "this is [name] speaking, how can I…"
    "my name is",
    "good morning",
    "good afternoon",
    "good evening",
    "i'll be happy to help",
    "i'd be happy to help",
    "happy to help you today",
]

_ANNOUNCEMENT_KWS_DEFAULT = ["record", "recorded", "recording",
                              "monitor", "monitored", "monitoring"]

# ---------------------------------------------------------------------------
# Broad phrase lists used for initial speaker identification
# (_score_speakers_by_phrases scans the full transcript with these)
# ---------------------------------------------------------------------------

# Phrases strongly diagnostic of the *customer* speaking.
_CUSTOMER_PHRASES_DEFAULT = [
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
]

# Phrases that are strongly diagnostic of the *agent* speaking.
_AGENT_PHRASES_DEFAULT = [
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
    "before i process that", "before i do that", "before i go ahead",
    "before i can process", "before i can cancel", "before i cancel",
    "before i disconnect", "before we proceed",
    "let me see what i can offer", "let me see what offers",
    "let me see what i can do for you", "let me see what i can do",
    "see what i can do",
    "what i can offer",
    "what we can offer",
    "let me pull up some offers", "let me check what deals",
    "i have some great offers", "we have a great offer",
    "i can offer you", "what if i offered",
    "i'd hate to lose you", "i'd hate to see you go",
    "hate to lose you",
    "hate to see you go",
    "we value you as a customer",
    "instead of cancelling", "instead of disconnecting",
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
]


# ---------------------------------------------------------------------------
# Exclusive phrase lists used by _relabel_turns_by_content
#
# These phrases are SO role-specific that their presence in a turn attributed
# to the *wrong* speaker is almost certainly a PyAnnote word-assignment error.
# Coverage spans ALL major call-centre call types:
#   • Technical support (ISP / software / hardware)
#   • Billing & payments
#   • Cancellation / retention
#   • General service enquiries
#   • Returns & refunds (e-commerce / retail)
#   • Complaints & escalations
# ---------------------------------------------------------------------------

_AGENT_EXCLUSIVE_PHRASES_DEFAULT = [

    # ── Opening / greeting ────────────────────────────────────────────────
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

    # ── Account verification (agent reads or verifies) ─────────────────────
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

    # ── Account observations (agent reads system) ──────────────────────────
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

    # ── Hold / transfer ────────────────────────────────────────────────────
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

    # ── Notes / account updates ────────────────────────────────────────────
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

    # ── Processing requests ────────────────────────────────────────────────
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

    # ── Technical support (agent actions) ─────────────────────────────────
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

    # ── Billing actions (agent initiates) ─────────────────────────────────
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

    # ── Offers / retention / upsell ────────────────────────────────────────
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

    # ── Understanding / retention framing (agent's perspective only) ───────
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

    # ── Empathy / professional courtesy (agent framing — one-directional) ──
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

    # ── Questions only an agent asks ───────────────────────────────────────
    "what seems to be the issue",
    "what can i help you with",
    "what is the issue today",
    "can you describe the issue",
    "can you describe the problem",
    "is there anything i can do",
    "anything i can do to keep",
    "happy to help you with that",

    # ── Closing ────────────────────────────────────────────────────────────
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
]

_CUSTOMER_EXCLUSIVE_PHRASES_DEFAULT = [

    # ── Cancellation / disconnection requests ──────────────────────────────
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

    # ── Refund / return requests ───────────────────────────────────────────
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

    # ── Billing complaints (customer's direct experience) ─────────────────
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

    # ── Rejecting retention offers ─────────────────────────────────────────
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

    # ── Escalation requests ────────────────────────────────────────────────
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

    # ── Frustration / complaint expressions ───────────────────────────────
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

    # ── Service / product problems (customer's lived experience) ──────────
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

    # ── Tenure / loyalty expressions ───────────────────────────────────────
    "i've been with you for",
    "i've been a customer for",
    "i've been your customer",
    "i'm a long-time customer",
    "after all these years",
    "i've been loyal",
    "i've been paying for",

    # ── Decision finality / direct demands ────────────────────────────────
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
]


def _relabel_turns_by_content(turns: list) -> list:
    """
    Post-process role labels using the *text content* of each turn.

    When PyAnnote mis-assigns a word's time window to the wrong speaker, the
    resulting turn will contain phrases that contradict its assigned role.
    We detect this by checking role-exclusive phrase hits: if a 'Customer' turn
    contains agent-exclusive language (or vice versa), we flip the role label.

    Scoring rules:
    - When only ONE side has any hits → always flip if the current label is wrong.
    - When BOTH sides have hits (genuinely mixed / merged segment) → require the
      dominant side to lead by at least 2 hits before flipping.  This prevents
      false flips on long turns that contain a few phrases from each speaker
      because PyAnnote merged two consecutive utterances into one segment.

    After relabeling, consecutive same-role turns are re-merged (excluding
    gap-split boundaries) so the final transcript looks natural.
    """
    agent_exclusive    = _pl.get("agent_exclusive_phrases",    _AGENT_EXCLUSIVE_PHRASES_DEFAULT)
    customer_exclusive = _pl.get("customer_exclusive_phrases", _CUSTOMER_EXCLUSIVE_PHRASES_DEFAULT)

    relabeled = []
    for turn in turns:
        t = dict(turn)  # shallow copy — don't mutate the input
        role = t.get("role", "")
        text_lower = t.get("text", "").lower()

        agent_excl_hits    = sum(1 for p in agent_exclusive    if p in text_lower)
        customer_excl_hits = sum(1 for p in customer_exclusive if p in text_lower)

        # Require a larger margin when both sides have hits — that indicates a
        # merged turn rather than a simple mis-assignment.
        both_active = agent_excl_hits > 0 and customer_excl_hits > 0
        margin_needed = 2 if both_active else 1

        flip_to_agent    = role == "Customer" and (agent_excl_hits - customer_excl_hits) >= margin_needed
        flip_to_customer = role == "Agent"    and (customer_excl_hits - agent_excl_hits) >= margin_needed

        if flip_to_agent:
            t["role"] = "Agent"
            print(
                f"[Whisperer] Content relabel Customer→Agent "
                f"(agent_excl={agent_excl_hits}, customer_excl={customer_excl_hits}, "
                f"margin={margin_needed}): \"{t['text'][:80]}\"",
                file=sys.stderr, flush=True
            )
        elif flip_to_customer:
            t["role"] = "Customer"
            print(
                f"[Whisperer] Content relabel Agent→Customer "
                f"(customer_excl={customer_excl_hits}, agent_excl={agent_excl_hits}, "
                f"margin={margin_needed}): \"{t['text'][:80]}\"",
                file=sys.stderr, flush=True
            )

        relabeled.append(t)

    # Re-merge consecutive same-role turns produced by relabeling.
    # Still respects gap-split boundaries (forced speaker-change markers).
    merged: list[dict] = []
    for t in relabeled:
        if (merged and
                merged[-1]["role"] == t["role"] and
                not t.get("_gap_split")):
            prev = merged[-1]
            prev["end"] = max(prev["end"], t["end"])
            if prev["text"] and not prev["text"].endswith(" "):
                prev["text"] += " "
            prev["text"] += t["text"]
        else:
            merged.append(t)
    return merged


def _detect_agent_by_greeting(utterances: list) -> str | None:
    """
    Return the speaker ID of the first utterance that contains one of the
    canonical agent greeting phrases.  Returns None if no greeting is found.
    Examines the first 8 utterances maximum (the greeting is always near the
    start of the call).
    """
    phrases = _pl.get("agent_greeting_phrases", _AGENT_GREETING_PHRASES_DEFAULT)
    for u in sorted(utterances, key=lambda x: x["start"])[:8]:
        text_lower = u.get("text", "").lower()
        for phrase in phrases:
            if phrase in text_lower:
                return u["speaker"]
    return None


def _score_speakers_by_phrases(utterances: list) -> str | None:
    """
    Count agent-diagnostic and customer-diagnostic phrase hits across ALL
    utterances for each speaker, then return the speaker with the highest net
    agent score.  Agent phrases are weighted +2; customer phrases are weighted -1.

    Scanning the full transcript (instead of just the first N utterances) makes
    the heuristic robust to calls that begin mid-conversation and to cases where
    PyAnnote mis-segments the early words — a single wrong assignment won't flip
    the result when many correct assignments are also counted.

    Returns None if the result is a tie (inconclusive).
    """
    speakers = list({u["speaker"] for u in utterances})
    if len(speakers) < 2:
        return None

    agent_phrases    = _pl.get("agent_phrases",    _AGENT_PHRASES_DEFAULT)
    customer_phrases = _pl.get("customer_phrases", _CUSTOMER_PHRASES_DEFAULT)

    agent_score: dict[str, int] = {s: 0 for s in speakers}
    for u in utterances:
        spk = u["speaker"]
        text_lower = u.get("text", "").lower()
        for phrase in agent_phrases:
            if phrase in text_lower:
                agent_score[spk] += 2
        for phrase in customer_phrases:
            if phrase in text_lower:
                agent_score[spk] -= 1

    best  = max(speakers, key=lambda s: agent_score[s])
    worst = min(speakers, key=lambda s: agent_score[s])
    if agent_score[best] == agent_score[worst]:
        return None  # tie — inconclusive, fall through to next heuristic

    print(
        f"[Whisperer] Phrase scores: {agent_score} → agent={best}",
        file=sys.stderr, flush=True
    )
    return best


def _detect_agent_by_word_count(utterances: list) -> str | None:
    """
    In a customer-support call the agent typically speaks more words in total
    than the customer.  Use this as a tie-breaker / last-resort heuristic.
    """
    word_counts: dict[str, int] = {}
    for u in utterances:
        spk = u["speaker"]
        word_counts[spk] = word_counts.get(spk, 0) + len(u.get("text", "").split())
    if not word_counts:
        return None
    return max(word_counts, key=word_counts.get)


def _validate_agent_assignment(utterances: list, agent_speaker: str) -> str:
    """
    Post-assignment sanity check: if the selected 'agent' speaker has
    significantly more customer-diagnostic phrases than the actual customer
    speaker, the labels are likely backwards and we swap them.

    This catches the failure mode where PyAnnote mis-assigns words at the
    segment boundary and our heuristics fire on mis-labelled text.
    """
    speakers = list({u["speaker"] for u in utterances})
    customer_speaker = next((s for s in speakers if s != agent_speaker), None)
    if not customer_speaker:
        return agent_speaker

    cust_phrases = _pl.get("customer_phrases", _CUSTOMER_PHRASES_DEFAULT)

    def _count_customer_phrases(spk: str) -> int:
        text = " ".join(u.get("text", "") for u in utterances if u["speaker"] == spk).lower()
        return sum(1 for p in cust_phrases if p in text)

    agent_cust_hits = _count_customer_phrases(agent_speaker)
    cust_cust_hits  = _count_customer_phrases(customer_speaker)

    if agent_cust_hits > 2 and agent_cust_hits > cust_cust_hits:
        print(
            f"[Whisperer] VALIDATION SWAP: '{agent_speaker}' had {agent_cust_hits} customer phrases "
            f"vs '{customer_speaker}' with {cust_cust_hits} — swapping agent assignment",
            file=sys.stderr, flush=True
        )
        return customer_speaker  # the other speaker is the real agent

    return agent_speaker


def _pick_agent_speaker(utterances: list, agent_hint: str | None = None) -> str:
    """
    Unified agent-speaker selection used by both mono-diarization and stereo
    transcription paths.

    Priority order:
    1. Explicit hint (already confirmed by caller to be the agent channel/ID).
    2. Greeting phrase detection — canonical agent opening phrases (first 8 utterances).
    3. Full-transcript phrase scoring — counts agent/customer diagnostic phrases
       across ALL utterances; most robust against partial mis-segmentation.
    4. Speaker with the most total words (agents typically speak more).
    5. First substantial non-announcement utterance (last-resort; risky for
       mid-recording calls where the customer may speak first).
    6. Earliest speaker (absolute fallback).
    """
    # Log word counts and first utterance per speaker for diagnostics
    word_counts: dict[str, int] = {}
    first_texts: dict[str, str] = {}
    for u in sorted(utterances, key=lambda x: x["start"]):
        spk = u["speaker"]
        word_counts[spk] = word_counts.get(spk, 0) + len(u.get("text", "").split())
        if spk not in first_texts:
            first_texts[spk] = u.get("text", "")[:80]

    print(
        f"[Whisperer] _pick_agent_speaker: word_counts={word_counts}",
        file=sys.stderr, flush=True
    )
    for spk, text in first_texts.items():
        print(
            f"[Whisperer]   first utterance {spk}: \"{text}\"",
            file=sys.stderr, flush=True
        )

    # 1 — explicit hint
    if agent_hint and any(u["speaker"] == agent_hint for u in utterances):
        print(f"[Whisperer] Agent identified by hint: {agent_hint}", file=sys.stderr, flush=True)
        return agent_hint

    # 2 — greeting phrase detection
    by_greeting = _detect_agent_by_greeting(utterances)
    if by_greeting is not None:
        print(f"[Whisperer] Agent identified by greeting: {by_greeting}", file=sys.stderr, flush=True)
        return by_greeting

    # 3 — full-transcript phrase scoring
    by_phrases = _score_speakers_by_phrases(utterances)
    if by_phrases is not None:
        print(f"[Whisperer] Agent identified by phrase scoring: {by_phrases}", file=sys.stderr, flush=True)
        return by_phrases

    # 4 — most words (before first-substantial-speaker to avoid misidentifying
    #     the customer as agent when a call starts mid-conversation)
    by_words = _detect_agent_by_word_count(utterances)
    if by_words is not None:
        print(f"[Whisperer] Agent identified by word count: {by_words} (counts: {word_counts})", file=sys.stderr, flush=True)
        return by_words

    # 5 — first substantial non-announcement utterance
    announcement_kws = frozenset(_pl.get("announcement_keywords", _ANNOUNCEMENT_KWS_DEFAULT))
    for u in sorted(utterances, key=lambda x: x["start"]):
        words = [w.strip(".,!?;:\"'") for w in u.get("text", "").lower().split()]
        if announcement_kws.intersection(words):
            continue
        if len(words) >= 3:
            print(f"[Whisperer] Agent identified by first substantial utterance: {u['speaker']}", file=sys.stderr, flush=True)
            return u["speaker"]

    # 6 — absolute fallback
    if utterances:
        fallback = min(utterances, key=lambda u: u["start"])["speaker"]
        print(f"[Whisperer] Agent identified by fallback (earliest): {fallback}", file=sys.stderr, flush=True)
        return fallback
    return "SPEAKER_00"

dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Module-level caches — models are loaded once per gunicorn worker process and
# reused across all requests.  This eliminates the 20–30 s per-call overhead
# and prevents memory pressure that causes pyannote to fail silently.
_whisper_model_cache: dict = {}
_pyannote_pipeline_cache = None

if not HF_TOKEN:
    print(
        "[Whisperer] CRITICAL: HF_TOKEN is not set — "
        "PyAnnote speaker diarization is DISABLED. "
        "All transcripts will use the low-quality silence-gap fallback. "
        "Set HF_TOKEN in /opt/agentsights/.env and restart the container.",
        file=sys.stderr, flush=True
    )
elif not HF_TOKEN.strip('"').strip("'").startswith("hf_"):
    print(
        f"[Whisperer] CRITICAL: HF_TOKEN value does not start with 'hf_' — "
        "PyAnnote authentication will fail.",
        file=sys.stderr, flush=True
    )
else:
    print(
        f"[Whisperer] HF_TOKEN loaded (hf_...{HF_TOKEN.strip(chr(34)).strip(chr(39))[-4:]})",
        file=sys.stderr, flush=True
    )


def _get_whisper_model(model_size: str, device: str, compute_type: str):
    """Return a cached WhisperX ASR model, loading it on first use."""
    import whisperx
    key = (model_size, device, compute_type)
    if key not in _whisper_model_cache:
        print(f"[Whisperer] Loading WhisperX '{model_size}' model (first use) …",
              file=sys.stderr, flush=True)
        _whisper_model_cache[key] = whisperx.load_model(model_size, device, compute_type=compute_type)
        print(f"[Whisperer] ✓ WhisperX '{model_size}' model cached",
              file=sys.stderr, flush=True)
    return _whisper_model_cache[key]


def _get_pyannote_pipeline(token: str):
    """Return a cached pyannote diarization pipeline, loading it on first use."""
    global _pyannote_pipeline_cache
    if _pyannote_pipeline_cache is None:
        from pyannote.audio import Pipeline
        print("[Whisperer] Loading pyannote/speaker-diarization-3.1 (first use) …",
              file=sys.stderr, flush=True)
        try:
            _pyannote_pipeline_cache = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token,
            )
        except TypeError:
            _pyannote_pipeline_cache = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=token,
            )
        print("[Whisperer] ✓ PyAnnote pipeline cached",
              file=sys.stderr, flush=True)
    return _pyannote_pipeline_cache


def _detect_audio_properties(audio_path: str):
    try:
        data, sr = sf.read(audio_path)
        is_stereo = data.ndim > 1 and data.shape[1] >= 2

        if is_stereo:
            import numpy as np
            left  = data[:, 0]
            right = data[:, 1]
            # If both channels are nearly identical, it's mono saved as stereo.
            # Use channel correlation: >0.99 means the same audio on both tracks.
            correlation = float(np.corrcoef(left, right)[0, 1])
            if correlation > 0.99:
                return False, None  # treat as mono → use diarization path
            return True, 2
        else:
            return False, None

    except Exception as e:
        return False, None


def _convert_pyannote_to_whisperx(pyannote_result):
    # pyannote.audio < 3.3 returns a pyannote.core.Annotation directly (has itertracks).
    # pyannote.audio >= 3.3 wraps it in a DiarizeOutput / similar dataclass.
    # Unwrap to get the Annotation regardless of version.
    annotation = pyannote_result
    if not hasattr(pyannote_result, 'itertracks'):
        for attr in ('speaker_diarization', 'exclusive_speaker_diarization', 'annotation', 'diarization', 'output', 'result'):
            candidate = getattr(pyannote_result, attr, None)
            if candidate is not None and hasattr(candidate, 'itertracks'):
                annotation = candidate
                break
        else:
            attrs = [a for a in dir(pyannote_result) if not a.startswith('_')]
            raise AttributeError(
                f"Cannot extract Annotation from pyannote output type "
                f"'{type(pyannote_result).__name__}'. "
                f"Available attributes: {attrs}"
            )

    segments = []
    for segment, track, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            'start': segment.start,
            'end': segment.end,
            'speaker': speaker
        })
    
    return {'segments': segments}

def _assign_speakers_to_words(diarize_segments, aligned):
    speaker_timeline = diarize_segments.get('segments', [])
    
    if not speaker_timeline:
        return aligned
    
    last_speaker = speaker_timeline[0].get('speaker', 'SPEAKER_00') if speaker_timeline else 'SPEAKER_00'

    for segment in aligned.get('segments', []):
        if 'words' not in segment or not segment['words']:
            continue

        prev_word_speaker = None

        for word in segment['words']:
            word_start = word.get('start', 0)
            word_end = word.get('end', 0)
            word_mid = (word_start + word_end) / 2.0

            best_speaker = None
            best_overlap = 0.0

            for spk_seg in speaker_timeline:
                spk_start = spk_seg.get('start', 0)
                spk_end = spk_seg.get('end', 0)

                overlap_start = max(word_start, spk_start)
                overlap_end = min(word_end, spk_end)
                overlap = max(0.0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = spk_seg.get('speaker', 'SPEAKER_00')

            if best_overlap == 0.0:
                # No time-overlap: prefer the segment that *contains* the word midpoint.
                # If none contains it, fall back to the previous word's speaker so we
                # don't incorrectly flip speakers mid-sentence.
                container = next(
                    (s for s in speaker_timeline
                     if s.get('start', 0) <= word_mid <= s.get('end', 0)),
                    None
                )
                if container:
                    best_speaker = container.get('speaker', 'SPEAKER_00')
                elif prev_word_speaker is not None:
                    # Same segment → very likely the same speaker continuing
                    best_speaker = prev_word_speaker
                else:
                    best_speaker = last_speaker

            word['speaker'] = best_speaker
            prev_word_speaker = best_speaker
            last_speaker = best_speaker

    return aligned


def transcribe_mono_with_diarization(
    audio_path: str,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    num_speakers: int | None = None,        
    agent_hint: str | None = None   
):
    import whisperx
    import os

    if device not in ["cpu", "cuda"]:
        device = "cpu"

    asr_model = _get_whisper_model(model_size, device, compute_type)
    batch_size = 16 if device != "cpu" else 4
    asr_result = asr_model.transcribe(audio_path, batch_size=batch_size)
    audio = whisperx.load_audio(audio_path)
    align_model, metadata = whisperx.load_align_model(asr_result["language"], device)
    aligned = whisperx.align(asr_result["segments"], align_model, metadata, audio, device)
    
    hf_token = HF_TOKEN
    clean_token = hf_token.strip('"').strip("'") if hf_token else None
    
    diarized = None
    diarization_failed = False

    # Pass the already-loaded waveform to pyannote as a dict instead of a file
    # path. This bypasses pyannote's internal file-loading code which uses
    # torchcodec.AudioDecoder — unavailable on CPU-only PyTorch builds.
    import torch
    waveform = torch.from_numpy(audio).float().unsqueeze(0)  # (1, samples)
    pyannote_input = {"waveform": waveform, "sample_rate": 16000}

    try:
        if not clean_token:
            raise RuntimeError("HF_TOKEN is not set — cannot load pyannote model")

        diarize_pipeline = _get_pyannote_pipeline(clean_token)

        print("[Whisperer] Running speaker diarization …",
              file=sys.stderr, flush=True)

        if num_speakers is not None:
            diarize_result = diarize_pipeline(pyannote_input, num_speakers=num_speakers)
        else:
            diarize_result = diarize_pipeline(pyannote_input)
        diarize_segments = _convert_pyannote_to_whisperx(diarize_result)
        diarized = _assign_speakers_to_words(diarize_segments, aligned)

        print("[Whisperer] ✓ PyAnnote diarization complete",
              file=sys.stderr, flush=True)

    except Exception as e:
        import traceback
        print(
            f"[Whisperer] DIARIZATION FAILED — falling back to low-quality synthetic method.\n"
            f"  Error type : {type(e).__name__}\n"
            f"  Error      : {e}\n"
            f"  Traceback  :\n{traceback.format_exc()}\n"
            f"\n"
            f"  ACTIONS TO FIX:\n"
            f"  1. Ensure HF_TOKEN is set in /opt/agentsights/.env (starts with hf_)\n"
            f"  2. Accept model licence at: https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            f"  3. Also accept: https://huggingface.co/pyannote/segmentation-3.0\n"
            f"  The HuggingFace account the token belongs to MUST be the account\n"
            f"  that accepted the licence.",
            file=sys.stderr, flush=True
        )
        diarization_failed = True

    diarization_method = "pyannote"
    if diarization_failed or diarized is None:
        print("[Whisperer] CRITICAL WARNING: Falling back to synthetic diarization "
              "(silence-gap method) — transcript speaker labels will be WRONG. "
              "See error above for how to fix PyAnnote authentication.",
              file=sys.stderr, flush=True)
        diarized = _create_synthetic_diarization(aligned)
        diarization_method = "synthetic_fallback"

    # Silence longer than this between consecutive same-speaker words strongly
    # suggests PyAnnote missed a speaker change at that pause.  We force a new
    # utterance at that boundary and mark it so the continuity pass below does
    # NOT re-merge it — keeping the split visible for downstream analysis.
    # 0.6 s is aggressive enough to catch most inter-speaker gaps while still
    # being larger than natural intra-sentence pauses (~0.1–0.3 s).
    SPEAKER_GAP_THRESHOLD = 0.6  # seconds

    utterances = []
    current = None
    prev_word_end = 0.0

    for segment in diarized.get("segments", []):
        if 'words' not in segment or not segment['words']:
            continue

        for word in segment['words']:
            spk = word.get('speaker', 'SPEAKER_00')
            word_start = word.get('start', 0)
            word_end = word.get('end', 0)
            word_text = word.get('word', '').strip()

            if not word_text:
                continue

            # Force a break when consecutive same-speaker words have a large
            # silence between them — PyAnnote likely missed the turn change here.
            same_speaker_gap = (
                current is not None and
                spk == current["speaker"] and
                prev_word_end > 0 and
                word_start - prev_word_end > SPEAKER_GAP_THRESHOLD
            )

            if current is None or spk != current["speaker"] or same_speaker_gap:
                if current:
                    current["text"] = current["text"].strip()
                    if current["text"]:
                        utterances.append(current)

                current = {
                    "speaker": spk,
                    "start": word_start,
                    "end": word_end,
                    "text": word_text,
                    "_gap_split": same_speaker_gap,  # preserve boundary marker
                }
            else:
                current["end"] = word_end
                if current["text"] and not current["text"].endswith(" "):
                    current["text"] += " "
                current["text"] += word_text

            prev_word_end = word_end

    if current:
        current["text"] = current["text"].strip()
        if current["text"]:
            utterances.append(current)

    def _enforce_speaker_continuity(utterances_list):
        # Merge consecutive same-speaker fragments, but NEVER merge across a
        # gap-split boundary — those represent suspected missed speaker changes.
        if not utterances_list:
            return utterances_list
        final = []
        for u in utterances_list:
            if (final and
                    final[-1]['speaker'] == u['speaker'] and
                    not u.get('_gap_split')):
                prev = final[-1]
                prev['end'] = max(prev['end'], u['end'])
                if prev.get('text') and not prev['text'].endswith(' '):
                    prev['text'] += ' '
                prev['text'] += u.get('text', '')
            else:
                final.append(u)
        return final

    utterances = _enforce_speaker_continuity(utterances)

    speakers = sorted(set(u["speaker"] for u in utterances))
    agent_speaker = _pick_agent_speaker(utterances, agent_hint)
    agent_speaker = _validate_agent_assignment(utterances, agent_speaker)

    role_map = {agent_speaker: "Agent"}
    non_agent_speakers = [spk for spk in speakers if spk != agent_speaker]
    if len(non_agent_speakers) == 1:
        role_map[non_agent_speakers[0]] = "Customer"
    else:
        for i, spk in enumerate(non_agent_speakers, 1):
            role_map[spk] = f"Customer {i}"

    print(
        f"[Whisperer] Final role map: {role_map}",
        file=sys.stderr, flush=True
    )

    turns = []
    for u in utterances:
        turns.append({
            "role": role_map[u["speaker"]],
            "start": round(float(u["start"]), 2),
            "end": round(float(u["end"]), 2),
            "text": u["text"],
            "_gap_split": u.get("_gap_split", False),
        })
    turns.sort(key=lambda x: x["start"])

    # Merge consecutive same-role turns, but NEVER across a gap-split boundary.
    # Gap splits mark suspected missed speaker changes; re-merging them would undo
    # the entire point of the split.
    merged_turns = []
    for turn in turns:
        if (merged_turns and
                merged_turns[-1]["role"] == turn["role"] and
                not turn.get("_gap_split")):
            prev = merged_turns[-1]
            prev["end"] = max(prev["end"], turn["end"])
            if prev["text"] and not prev["text"].endswith(" "):
                prev["text"] += " "
            prev["text"] += turn["text"]
        else:
            merged_turns.append(turn)
    turns = merged_turns

    # Content-based relabeling: fix turns where PyAnnote word-assignment errors
    # caused agent phrases to land in a Customer turn (or customer phrases in an
    # Agent turn).  The _gap_split flag is preserved through the relabeling so
    # the re-merge inside _relabel_turns_by_content still respects forced splits.
    turns = _relabel_turns_by_content(turns)

    # Remove internal flag before returning — not part of the public API
    for t in turns:
        t.pop("_gap_split", None)

    duration = 0.0
    if turns:
        duration = max(duration, max(t["end"] for t in turns))

    return {
        "call_id": os.path.basename(audio_path),
        "duration_sec": round(duration, 2),
        "utterances": turns,
        "diarization_method": diarization_method,
    }


def _create_synthetic_diarization(aligned):
    # 0.5 s covers typical inter-turn pauses in phone conversations (0.2–0.6 s).
    # The old 1.5 s threshold was too conservative and caused entire conversation
    # blocks to be merged into a single speaker.
    MIN_SILENCE_GAP = 0.5

    segments = aligned.get("segments", [])
    if not segments:
        return aligned

    # Collect all words with timing info in order
    all_words = []
    for seg in segments:
        if "words" in seg and seg["words"]:
            all_words.extend(seg["words"])
        else:
            # Segment has no word-level timing – synthesise a single word entry
            text = seg.get("text", "").strip()
            if text:
                all_words.append({
                    "word": text,
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                })

    if not all_words:
        return aligned

    # Assign speakers based on silence gaps
    current_speaker = "SPEAKER_00"
    for i, word in enumerate(all_words):
        word["speaker"] = current_speaker
        if i < len(all_words) - 1:
            gap = all_words[i + 1].get("start", 0) - word.get("end", 0)
            if gap > MIN_SILENCE_GAP:
                current_speaker = "SPEAKER_01" if current_speaker == "SPEAKER_00" else "SPEAKER_00"

    # Re-pack words back into their original segments so the downstream
    # `for segment in diarized["segments"]: for word in segment["words"]` loop works.
    result = aligned.copy()
    new_segments = []
    word_idx = 0
    for seg in segments:
        new_seg = dict(seg)
        if "words" in seg and seg["words"]:
            count = len(seg["words"])
            new_seg["words"] = all_words[word_idx:word_idx + count]
            word_idx += count
        else:
            # Segments that had no words keep their synthetic single word
            if word_idx < len(all_words):
                new_seg["words"] = [all_words[word_idx]]
                word_idx += 1
        new_segments.append(new_seg)
    result["segments"] = new_segments
    return result


def transcribe_stereo_channels(
    audio_path: str,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    agent_hint: str | None = None,
):
    """
    Transcribe a stereo recording by splitting channels first.

    In call-centre recordings each channel typically carries a single speaker
    (e.g. left = agent, right = customer).  Transcribing the channels
    independently avoids diarization entirely, giving perfect speaker
    separation and better word accuracy (no mixed-speaker audio fed to ASR).

    The channel whose first utterance starts earliest is treated as the Agent,
    unless agent_hint is supplied.
    """
    import whisperx
    import tempfile
    import numpy as np

    data, sr = sf.read(audio_path)
    left  = data[:, 0]
    right = data[:, 1]

    results = {}
    for label, channel_data in [("ch0", left), ("ch1", right)]:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, channel_data, sr)
        tmp.close()
        try:
            asr_model = _get_whisper_model(model_size, device, compute_type)
            batch_size = 16 if device != "cpu" else 4
            asr_result = asr_model.transcribe(tmp.name, batch_size=batch_size)
            audio = whisperx.load_audio(tmp.name)
            align_model, metadata = whisperx.load_align_model(asr_result["language"], device)
            aligned = whisperx.align(asr_result["segments"], align_model, metadata, audio, device)
            results[label] = aligned
        finally:
            try:
                os.remove(tmp.name)
            except Exception:
                pass

    def _extract_utterances(aligned, speaker_label):
        # Each WhisperX segment is a natural phrase/sentence boundary.
        # Create one utterance per segment so the transcript is readable,
        # not one giant utterance for the whole call.
        utterances = []
        for segment in aligned.get("segments", []):
            words = segment.get("words", [])
            if words:
                # Build text from word list (more accurate timings)
                word_items = [w for w in words if w.get("word", "").strip()]
                if not word_items:
                    continue
                seg_start = word_items[0].get("start", segment.get("start", 0))
                seg_end   = word_items[-1].get("end",   segment.get("end",   0))
                text = " ".join(w["word"].strip() for w in word_items)
            else:
                text = segment.get("text", "").strip()
                if not text:
                    continue
                seg_start = segment.get("start", 0)
                seg_end   = segment.get("end",   0)
            utterances.append({
                "speaker": speaker_label,
                "start": seg_start,
                "end":   seg_end,
                "text":  text,
            })
        return utterances

    ch0_utterances = _extract_utterances(results.get("ch0", {}), "ch0")
    ch1_utterances = _extract_utterances(results.get("ch1", {}), "ch1")

    # Determine which channel is the Agent using the unified multi-heuristic
    # picker (greeting phrases → first 3+ word non-announcement → most words).
    all_channel_utterances = ch0_utterances + ch1_utterances
    agent_ch = _pick_agent_speaker(all_channel_utterances, agent_hint if agent_hint in ("ch0", "ch1") else None)

    role_map = {agent_ch: "Agent", ("ch1" if agent_ch == "ch0" else "ch0"): "Customer"}

    all_utterances = ch0_utterances + ch1_utterances
    all_utterances.sort(key=lambda u: u["start"])

    turns = [
        {
            "role": role_map[u["speaker"]],
            "start": round(float(u["start"]), 2),
            "end": round(float(u["end"]), 2),
            "text": u["text"],
        }
        for u in all_utterances
    ]

    # Consolidate consecutive same-role turns
    merged_turns = []
    for turn in turns:
        if merged_turns and merged_turns[-1]["role"] == turn["role"]:
            prev = merged_turns[-1]
            prev["end"] = max(prev["end"], turn["end"])
            if not prev["text"].endswith(" "):
                prev["text"] += " "
            prev["text"] += turn["text"]
        else:
            merged_turns.append(turn)

    duration = max((t["end"] for t in merged_turns), default=0.0)
    return {
        "call_id": os.path.basename(audio_path),
        "duration_sec": round(duration, 2),
        "utterances": merged_turns,
        "diarization_method": "stereo_channels",
    }


def transcribe_audio(
    audio_path: str,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    agent_hint: str | None = None
):
    if device == "mps":
        device = "cpu"

    print(f"[Whisperer] transcribe_audio called: model={model_size}, device={device}, "
          f"compute={compute_type}, file={os.path.basename(audio_path)}",
          file=sys.stderr, flush=True)

    is_stereo, estimated_speakers = _detect_audio_properties(audio_path)

    if is_stereo:
        print("[Whisperer] Stereo audio detected — using per-channel transcription",
              file=sys.stderr, flush=True)
        # Stereo recordings have one speaker per channel — transcribe each
        # channel independently for perfect speaker separation.
        try:
            result = transcribe_stereo_channels(
                audio_path=audio_path,
                model_size=model_size,
                device=device,
                compute_type=compute_type,
                agent_hint=agent_hint,
            )
            print(f"[Whisperer] ✓ Stereo transcription complete: "
                  f"{len(result.get('utterances', []))} utterances, "
                  f"method={result.get('diarization_method')}",
                  file=sys.stderr, flush=True)
            return result
        except Exception as exc:
            print(f"[Whisperer] Stereo transcription failed ({type(exc).__name__}: {exc}) — "
                  f"falling through to mono+diarization",
                  file=sys.stderr, flush=True)

    print("[Whisperer] Mono audio — using pyannote diarization path",
          file=sys.stderr, flush=True)

    num_speakers = estimated_speakers if estimated_speakers is not None else 2
    result = transcribe_mono_with_diarization(
        audio_path=audio_path,
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        num_speakers=num_speakers,
        agent_hint=agent_hint
    )
    print(f"[Whisperer] ✓ Mono transcription complete: "
          f"{len(result.get('utterances', []))} utterances, "
          f"method={result.get('diarization_method')}",
          file=sys.stderr, flush=True)
    return result