from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

@dataclass
class Turn:
    turn_id: int
    speaker: str 
    start_sec: float
    end_sec: float
    text: str
    
    duration_sec: float = 0.0
    word_count: int = 0
    wpm: float = 0.0
    
    emotion: Optional[str] = None 
    emotion_score: Optional[float] = None
    
    contains_empathy: bool = False
    contains_apology: bool = False
    contains_escalation: bool = False
    
    sentiment: Optional[str] = None 
    sentiment_score: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
