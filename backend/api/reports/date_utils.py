from datetime import datetime, timedelta
from django.utils import timezone


def get_week_date_range(date=None):
    """
    Get the start and end date for a week (Monday to Sunday).
    If date is None, uses current date.
    """
    if date is None:
        date = timezone.now().date()
    
    # Get Monday of the week
    start_date = date - timedelta(days=date.weekday())
    # Get Sunday of the week
    end_date = start_date + timedelta(days=6)
    
    return start_date, end_date


def get_month_date_range(date=None):
    """
    Get the start and end date for a month.
    If date is None, uses current date.
    """
    if date is None:
        date = timezone.now().date()
    
    # First day of the month
    start_date = date.replace(day=1)
    
    # Last day of the month
    if date.month == 12:
        end_date = date.replace(year=date.year + 1, month=1, day=1) - timedelta(days=1)
    else:
        end_date = date.replace(month=date.month + 1, day=1) - timedelta(days=1)
    
    return start_date, end_date


def get_previous_week_date_range(date=None):
    """
    Get the start and end date for the previous week.
    """
    if date is None:
        date = timezone.now().date()
    
    # Go back 7 days
    previous_week_date = date - timedelta(days=7)
    return get_week_date_range(previous_week_date)


def get_previous_month_date_range(date=None):
    """
    Get the start and end date for the previous month.
    """
    if date is None:
        date = timezone.now().date()
    
    # First day of current month
    first_of_month = date.replace(day=1)
    # Last day of previous month
    last_of_previous_month = first_of_month - timedelta(days=1)
    
    return get_month_date_range(last_of_previous_month)


def get_weeks_in_month(start_date, end_date):
    """
    Get all week date ranges that fall within the given month range.
    Returns list of (week_start, week_end) tuples.
    """
    weeks = []
    current_date = start_date
    
    while current_date <= end_date:
        week_start, week_end = get_week_date_range(current_date)
        
        # Check if week overlaps with the month
        if week_end >= start_date and week_start <= end_date:
            # Trim to month boundaries
            actual_start = max(week_start, start_date)
            actual_end = min(week_end, end_date)
            weeks.append((actual_start, actual_end))
        
        # Move to next week
        current_date = week_end + timedelta(days=1)
    
    return weeks
