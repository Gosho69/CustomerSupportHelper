from django.core.management.base import BaseCommand
from django.utils import timezone
from reports.models import PerformanceReport
from reports.performance_calculator import PerformanceCalculator
from reports.date_utils import get_previous_month_date_range
from users.models import MyUser


class Command(BaseCommand):
    help = 'Generate monthly performance reports for all active agents'

    def add_arguments(self, parser):
        parser.add_argument(
            '--agent-id',
            type=int,
            help='Generate report for specific agent only',
        )

    def handle(self, *args, **options):
        start_date, end_date = get_previous_month_date_range()
        
        self.stdout.write(f'Generating monthly reports for period: {start_date} to {end_date}')
        
        if options['agent_id']:
            agents = MyUser.objects.filter(id=options['agent_id'], role='agent', is_active=True)
        else:
            agents = MyUser.objects.filter(role='agent', is_active=True)
        
        generated_count = 0
        skipped_count = 0
        error_count = 0
        
        for agent in agents:
            try:
                existing = PerformanceReport.objects.filter(
                    agent=agent,
                    report_type='monthly',
                    start_date=start_date,
                    end_date=end_date
                ).exists()
                
                if existing:
                    self.stdout.write(self.style.WARNING(
                        f'Skipping {agent.username} - report already exists'
                    ))
                    skipped_count += 1
                    continue
                
                calculator = PerformanceCalculator(agent, start_date, end_date, report_type='monthly')
                metrics = calculator.calculate_metrics()
                
                report = PerformanceReport.objects.create(
                    agent=agent,
                    report_type='monthly',
                    start_date=start_date,
                    end_date=end_date,
                    total_calls=metrics['call_volume']['total_calls'],
                    average_call_duration=metrics['call_volume']['average_duration'],
                    average_emotional_score=metrics['emotional_metrics']['average_score'],
                    positive_calls_percentage=metrics['emotional_metrics']['positive_percentage'],
                    negative_calls_percentage=metrics['emotional_metrics']['negative_percentage'],
                    emotional_trend=metrics['emotional_metrics']['trend'],
                    average_behavioral_score=metrics['behavioral_metrics']['average_score'],
                    empathy_score=metrics['behavioral_metrics']['empathy'],
                    professionalism_score=metrics['behavioral_metrics']['professionalism'],
                    problem_solving_score=metrics['behavioral_metrics']['problem_solving'],
                    behavioral_trend=metrics['behavioral_metrics']['trend'],
                    most_common_topics=metrics['topic_metrics']['most_common_topics'],
                    performance_consistency_score=metrics['consistency']['consistency_score'],
                    variance_from_average=metrics['consistency']['variance'],
                    percentile_score=metrics['comparison']['percentile'],
                    strengths=metrics['assessment']['strengths'],
                    weaknesses=metrics['assessment']['weaknesses'],
                    recommendations=metrics['assessment']['recommendations'],
                    weekly_analysis=metrics.get('weekly_analysis'),
                    overall_rating=metrics['assessment']['overall_rating'],
                    summary=metrics['assessment']['summary'],
                    generated_by=None
                )
                
                if metrics.get('weekly_analysis'):
                    weeks_used = metrics['weekly_analysis'].get('total_weeks_analyzed', 0)
                    self.stdout.write(self.style.SUCCESS(
                        f'Generated monthly report for {agent.username} using {weeks_used} weekly reports (ID: {report.id})'
                    ))
                else:
                    self.stdout.write(self.style.SUCCESS(
                        f'Generated monthly report for {agent.username} from raw calls (ID: {report.id})'
                    ))
                generated_count += 1
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(
                    f'Error generating report for {agent.username}: {str(e)}'
                ))
                error_count += 1
        
        self.stdout.write(self.style.SUCCESS(
            f'\nMonthly report generation completed:'
            f'\n- Generated: {generated_count}'
            f'\n- Skipped: {skipped_count}'
            f'\n- Errors: {error_count}'
        ))
