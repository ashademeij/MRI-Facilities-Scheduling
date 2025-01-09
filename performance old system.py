import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Simulation parameters for patient scan durations and arrival rates
mean_scan_type1 = 0.4327  # Mean scan time for Type 1 patients (in hours), based on part 1
sd_scan_type1 = 0.0978  # Standard deviation for scan Type 1 durations, based on part 1
avg_calls_type1 = 379 / 23  # Total calls type 1/ total days (23 = (31 days in August - 8 weekend days)
avg_calls_type2 = 239 / 23  # Total calls type 2/ total days (23 = (31 days in August - 8 weekend days)
alpha_type2 = 12.5848  # Shape parameter for Type 2, based on part 1 (Gamma distribution)
beta_type2 = 0.0532  # Scale parameter for Type 2, based on part 2 (Gamma distribution)

# Slot times of MRI scans for both types
slot_time_type1 = 0.5833  # Scan duration (35min) for Type 1 patients (in hours)
slot_time_type2 = 1.0  # Scan duration (60min) for Type 2 patients (in hours)

# Working hours for scheduling
day_start = 8  # Start of working day 08:00
day_end = 17  # End of working day 17:00


def simulation_calls():
    # Generate a simulated schedule of calls over a period of 30 days
    records = []
    start_date = datetime(2025, 1, 1)  # Starting date for simulation (arbitrary)
    for day in range(30):
        date = start_date + timedelta(days=day)
        number_of_calls_type1 = np.random.poisson(avg_calls_type1)  # Random number of Type 1 calls
        number_of_calls_type2 = np.random.poisson(avg_calls_type2)  # Random number of Type 2 calls
        call_time_type1 = call_time_type2 = 8.0  # Starting call time 8:00

        # Schedule calls for Type 1 patients
        for _ in range(number_of_calls_type1):
            duration = max(0, np.random.normal(mean_scan_type1, sd_scan_type1))
            records.append([date.strftime('%Y-%m-%d'), call_time_type1, duration, 'Type 1'])
            call_time_type1 += np.random.exponential(1 / 1.833705)  # Time until next call

        # Schedule calls for Type 2 patients
        for _ in range(number_of_calls_type2):
            duration = max(0, np.random.gamma(alpha_type2, 1 / beta_type2))
            records.append([date.strftime('%Y-%m-%d'), call_time_type2, duration, 'Type 2'])
            call_time_type2 += max(0, np.random.normal(0.8666, 0.31078))  # Time until next call

    # Combine the simulated call schedule into a DataFrame
    calling_schedule = pd.DataFrame(records, columns=['Date', 'Time', 'Duration', 'PatientType'])
    calling_schedule.sort_values(by=['Date', 'Time'], inplace=True)
    calling_schedule['CallDateTime'] = pd.to_datetime(calling_schedule['Date']) + pd.to_timedelta(
        calling_schedule['Time'], unit='h')
    return calling_schedule


def rounding_up(dt):
    # Rounds up the time to the nearest 5 minutes, for logistic reasons
    minutes = dt.minute
    if minutes % 5 != 0:
        dt += timedelta(minutes=(5 - minutes % 5))
    return dt.replace(second=0, microsecond=0)

def get_available_slot(call_time, machine_available, scan_duration):
    # Determines the next available time slot for scheduling a scan on machines op both types
    earliest_next_scan = (call_time + timedelta(days=1)).replace(hour=day_start, minute=0)
    if machine_available < earliest_next_scan:
        machine_available = earliest_next_scan
    machine_available = rounding_up(machine_available)

    # Checks if the available slot is within operation hours
    if machine_available.time() < datetime.min.replace(hour=day_end).time():
        return max(earliest_next_scan, machine_available)  # Earliest available time
    # If not, schedule for the next available day
    return rounding_up(
        machine_available.replace(hour=day_start, minute=0) + timedelta(days=1))


def scans_schedule(df):
    # Schedule scans based on patient calls and machine availability
    mri1_available = mri2_available = datetime.min
    scheduled_type1 = []
    scheduled_type2 = []

    for _, row in df.iterrows():
        call_time = row['CallDateTime']
        if row['PatientType'] == 'Type 1':
            scheduled_time = get_available_slot(call_time, mri1_available, slot_time_type1)
            scheduled_type1.append((row['CallDateTime'], row['PatientType'], scheduled_time))
            mri1_available = scheduled_time + timedelta(hours=slot_time_type1)
        else:
            scheduled_time = get_available_slot(call_time, mri2_available, slot_time_type2)
            scheduled_type2.append((row['CallDateTime'], row['PatientType'], scheduled_time))
            mri2_available = scheduled_time + timedelta(hours=slot_time_type2)

    return (pd.DataFrame(scheduled_type1, columns=['CallDateTime', 'PatientType', 'ScheduledTime']),
            pd.DataFrame(scheduled_type2, columns=['CallDateTime', 'PatientType', 'ScheduledTime']))


def calculate_waitingtime_1(call_time, scheduled_time):
    # Calculates the waiting time for a patient from their call to their scheduled time
    waiting_time = 0
    while call_time < scheduled_time:
        next_hour = call_time.replace(minute=0, second=0) + timedelta(hours=1)
        if day_start <= call_time.hour < day_end:
            if scheduled_time < next_hour:
                waiting_time += (scheduled_time - call_time).total_seconds() / 3600  # Calculate waiting time
                break
            else:
                waiting_time += (next_hour - call_time).total_seconds() / 3600
        call_time = next_hour
        if call_time.hour >= day_end:
            call_time = call_time.replace(hour=day_start, minute=0) + timedelta(days=1)
    return waiting_time


def calculate_waitingtime_2(scheduled_dfs):
    # Computes average and maximum waiting times
    waiting_times = [calculate_waitingtime_1(row['CallDateTime'], row['ScheduledTime']) for df in scheduled_dfs for
                     _, row in df.iterrows()]
    return (sum(waiting_times) / len(waiting_times) if waiting_times else 0,
            max(waiting_times) if waiting_times else 0)


def calculate_idle_time(scheduled_df):
    # Computes the total idle time for an MRI machine
    last_scan_end = datetime.min
    total_idle_time_seconds = 0
    for _, row in scheduled_df.iterrows():
        scheduled_start = row['ScheduledTime']
        if last_scan_end != datetime.min and day_start <= last_scan_end.hour < day_end:
            idle_time = scheduled_start - last_scan_end
            total_idle_time_seconds += max(idle_time.total_seconds(), 0)
        last_scan_end = scheduled_start + timedelta(
            hours=slot_time_type1 if row['PatientType'] == 'Type 1' else slot_time_type2)

    operational_days = (scheduled_df['ScheduledTime'].dt.date.max() - scheduled_df[
        'ScheduledTime'].dt.date.min()).days + 1
    return (total_idle_time_seconds / 3600) / operational_days if operational_days > 0 else 0


def calculate_average_appointments(scheduled_dfs):
    # Calculates average number of appointments per day for both types
    combined_df = pd.concat(scheduled_dfs)
    combined_df['Date'] = combined_df['ScheduledTime'].dt.date
    daily_counts = combined_df.groupby(['Date', 'PatientType']).size().reset_index(name='Counts')
    return daily_counts.groupby('PatientType')['Counts'].mean()


def calculate_total_appointments(df):
    # Returns total number of appointments scheduled by patient type
    return df['PatientType'].value_counts()


def performance_scans(df):
    # Main analysis to derive statistics from the scheduled scan records
    scheduled_type1, scheduled_type2 = scans_schedule(df)

    avg_waiting_type1, max_waiting_type1 = calculate_waitingtime_2([scheduled_type1])
    avg_waiting_type2, max_waiting_type2 = calculate_waitingtime_2([scheduled_type2])

    idle_time_mri1 = calculate_idle_time(scheduled_type1)
    idle_time_mri2 = calculate_idle_time(scheduled_type2)

    avg_appointments = calculate_average_appointments([scheduled_type1, scheduled_type2])
    total_appointments = calculate_total_appointments(df)

    return {
        'avg_waiting_type1': avg_waiting_type1,
        'max_waiting_type1': max_waiting_type1,
        'avg_waiting_type2': avg_waiting_type2,
        'max_waiting_type2': max_waiting_type2,
        'idle_time_mri1': idle_time_mri1,
        'idle_time_mri2': idle_time_mri2,
        'avg_appointments': avg_appointments,
        'total_appointments': total_appointments
    }


# Main execution flow
simulated_df = simulation_calls()  # Step 1: Simulate the call schedule
results = performance_scans(simulated_df)  # Step 2: Analyze the appointments and waiting times

# Display a portion of the generated schedule
print("Part of the simulated schedule:")
print(simulated_df.head(20))

# Present the results of the analysis
print("\nAnalysis Results:")
for key, value in results.items():
    print(f"{key}: {value}")
