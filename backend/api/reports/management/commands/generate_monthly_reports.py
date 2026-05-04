from django.core.management.base import BaseCommand
from reports.models import PerformanceReport
from reports.report_factory import calculate_and_create_report
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
                # Never generate a report for a period that ended before the
                # agent was created — there cannot be any meaningful data there.
                if end_date < agent.created_at.date():
                    self.stdout.write(self.style.WARNING(
                        f'Skipping {agent.username} — period {start_date}–{end_date} '
                        f'predates account creation ({agent.created_at.date()})'
                    ))
                    skipped_count += 1
                    continue

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
                
                report = calculate_and_create_report(
                    agent=agent,
                    report_type='monthly',
                    start_date=start_date,
                    end_date=end_date,
                )
                
                self.stdout.write(self.style.SUCCESS(
                    f'Generated monthly report for {agent.username} (ID: {report.id})'
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
