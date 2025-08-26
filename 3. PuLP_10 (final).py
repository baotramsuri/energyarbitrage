import requests # fetch data from the Internet
from datetime import datetime, timedelta, time # handle date, time
import pytz # handle time zones
import pandas as pd # process data tables and perform data analysis
from entsoe import EntsoePandasClient #access electricity data
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus #solve linear programming optimization problems
import matplotlib.pyplot as plt #plot charts and graphs
import numpy as np #perform numerical operations and efficiently handle arrays


## ----- STEP 1: SYSTEM CONFIGURATION 

## 1. PV SOLAR
lat = 63.096 #Finland
lon = 21.616
year = 2023  # Use 2023 data, because PVgis only provide historical data of solar irradiance from 2023 backwards.
PV_rated = 20  # MW, Max power of PV system
PR = 0.8 # System efficiency


## 2. BESS
battery_capacity = 30  # MWh
charge_power_limit = 10  # MW
discharge_power_limit = 10  # MW
charge_eff = 0.9
discharge_eff = 0.9
initial_soc = 0.0 * battery_capacity
cycle_cost= 0 # €/MWh

## 3. GRID
grid_power_limit = 10  # MW

## 4. GRID fee
transmission_cost= 41.5 #Transmission cost 4.15c/kWh. Only apply for import
marginal_export=2 #Marginal price of selling to grid (no applied VAT) 0.2c/kWh
marginal_import=6 #Marginal price of buying from grid (included VAT) 0.6c/kWh
consumption_tax=27.9 #Consumption tax (included VAT) 2.79c/kWh
VAT_tax=0.24 #VAT 24%

## ----- STEP 2: GET THE D-A ELECTRICITY PRICE FROM ENTSO

# --- LOCAL TIMEZONE ---
T = 24 # Time range = 24 hours
helsinki_tz = pytz.timezone("Europe/Helsinki")
today = datetime.now(helsinki_tz).date()

# Get the desired date as local time
selected_date = today + timedelta(days=0) # User can set any date here
local_start = pd.Timestamp(datetime(selected_date.year, selected_date.month, selected_date.day), tz=helsinki_tz) # Choose Helsinki timezone
local_end   = local_start + pd.Timedelta(hours=24) #It makes 00:00 as default start time, we add 24 hours to cover full day

# Convert to UTC for ENTSO-E API
start_utc = local_start.tz_convert("UTC")
end_utc   = local_end.tz_convert("UTC")


# Query day-ahead prices between start_date and end_date (local time aware)
api_key = "  your key  "
client = EntsoePandasClient(api_key=api_key)
country_code = '10YFI-1--------U'  # Finland's country code
prices = client.query_day_ahead_prices(country_code, start=start_utc, end=end_utc)

# Convert back to local for filtering
prices_local = prices.tz_convert(helsinki_tz)
prices_local = prices_local[(prices_local.index >= local_start) & (prices_local.index < local_end)]

# Round and list
da_prices = [round(price, 2) for price in prices_local]

print(f"✅ Got {len(da_prices)} prices for {selected_date.strftime('%Y-%m-%d')} local time")
print(da_prices)

## ----- STEP 3: GET THE HOURLY SOLAR IRRADIANCE FROM PVGIS & CALCULATE THE PV GENERATED POWER

# --- Call PVgis API to get hourly solar irradiance data ---
def get_pvgis_hourly(lat, lon, year):
    url = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"
    params = {
        "lat": lat,
        "lon": lon,
        "startyear": year,
        "endyear": year,
        "usehorizon": 0,
        "outputformat": "json",
        "pvtechchoice": "crystSi",
        "mounting_system": "fixed",
        "angle": 70,  # slope
        "aspect": 0,  # azimuth
        "optimalinclination": 0
    }
    response = requests.get(url, params=params)
    data = response.json()
    if "outputs" not in data or "hourly" not in data["outputs"]:
        raise ValueError("API didn't return hourly data")
    return data["outputs"]["hourly"]

pvgis_data = get_pvgis_hourly(lat, lon, 2023)

# --- Get datetime from PVGIS data response ---
def parse_pvgis_time(h):
    raw = h["time"]  # e.g. "20230731:2211"
    date_part, hm_part = raw.split(':')
    dt = datetime.strptime(date_part + hm_part[:2], "%Y%m%d%H")
    return dt

# --- Filter to correct the selected date and month ---
filtered_data = [
    h for h in pvgis_data
    if parse_pvgis_time(h).month == selected_date.month
    and parse_pvgis_time(h).day == selected_date.day
]

print(f"✅ Got {len(filtered_data)} solar irradiance records for {selected_date.strftime('%Y-%m-%d')} (local time)")

# --- Calculate PV generated power ---
solar_irradiance = [h.get("G(i)", 0) for h in filtered_data]
pv_profile = [(G / 1000) * PV_rated * PR for G in solar_irradiance]

## ----- STEP 4.1: START PULP OPTIMIZING MODEL

from pulp import *

# --- Initialize model ---
prob = LpProblem("Energy_Optimization", LpMaximize)

# --- Decision variable ---
pv2batt = [LpVariable(f"pv2batt_{t}", lowBound=0) for t in range(T)]
pv2grid = [LpVariable(f"pv2grid_{t}", lowBound=0) for t in range(T)]
curtail = [LpVariable(f"curtail_{t}", lowBound=0) for t in range(T)]
grid2batt = [LpVariable(f"grid2batt_{t}", lowBound=0) for t in range(T)]
batt2grid = [LpVariable(f"batt2grid_{t}", lowBound=0) for t in range(T)]
soc = [LpVariable(f"soc_{t}", lowBound=0, upBound=battery_capacity) for t in range(T)]


# --- Binary variable to prevent simultaneous charging/discharging ---
mode = [LpVariable(f"mode_{t}", cat="Binary") for t in range(T)]  # 1 if discharging, 0 if charging

# --- Constraints ---
for t in range(T):
    # 1. Total PV generated power = charge to battery + sell to grid + cultail
    prob += pv2batt[t] + pv2grid[t] + curtail[t] == pv_profile[t]

    # 2. SoC is updated per hour, initial SOC is assumed
    if t == 0:
        prob += soc[t] == initial_soc
    else:
        prob += soc[t] == soc[t-1] + (pv2batt[t] + grid2batt[t]) * charge_eff - batt2grid[t] / discharge_eff

    # 3. Charging limit
    prob += pv2batt[t] + grid2batt[t] <= charge_power_limit

    # 4. Discharging limit
    prob += batt2grid[t] <= discharge_power_limit

    # 5. Do not charge + discharge at the same time (big-M method)
    prob += pv2batt[t] + grid2batt[t] <= charge_power_limit * (1 - mode[t])
    prob += batt2grid[t] <= discharge_power_limit * mode[t]

    # 6. Limit power import from grid (charging only)
    prob += grid2batt[t] <= grid_power_limit

    # 7. Limit the power exported to the grid (PV + batt)
    prob += pv2grid[t] + batt2grid[t] <= grid_power_limit

    # 8. Discharge amount <= SOC.
    if t == 0:
        prob += batt2grid[t] <= initial_soc*discharge_eff
    else:
        prob += batt2grid[t] <= soc[t-1]*discharge_eff

    # 9. At the end of the day, force the battery to discharge to initial SOC.
    prob += soc[T-1] == initial_soc


## --- Objective function: maximize arbitrage profit ---
profit = lpSum([
    (pv2grid[t] + batt2grid[t]) * (da_prices[t]-marginal_export) - grid2batt[t] * (da_prices[t]*(1+VAT_tax)+marginal_import+consumption_tax+transmission_cost) - batt2grid[t] * cycle_cost

    for t in range(T)
])

prob += profit

# --- Solve ---
prob.solve(pulp.PULP_CBC_CMD(msg=False))

## STEP 4.2: TRY VARIOUS BATTERY CAPACITIES FROM 5 → 50 MWh ---
results = []

for cap in range(5, 71, 5):  # from 5 to 70 MWh (5, 10, 15, 20...)

    # --- Create new model ---
    prob2 = LpProblem("Energy_Optimization", LpMaximize)

    # --- Variable declaration ---
    pv2batt2 = [LpVariable(f"pv2batt_{t}", lowBound=0) for t in range(T)]
    pv2grid2 = [LpVariable(f"pv2grid_{t}", lowBound=0) for t in range(T)]
    curtail2 = [LpVariable(f"curtail_{t}", lowBound=0) for t in range(T)]
    grid2batt2 = [LpVariable(f"grid2batt_{t}", lowBound=0) for t in range(T)]
    batt2grid2 = [LpVariable(f"batt2grid_{t}", lowBound=0) for t in range(T)]
    soc_optimize = [LpVariable(f"soc_optimize_{t}", lowBound=0, upBound=cap) for t in range(T)]
    mode = [LpVariable(f"mode_{t}", cat="Binary") for t in range(T)]

    # --- Constraints  ---
    for t in range(T):
        prob2 += pv2batt2[t] + pv2grid2[t] + curtail2[t] == pv_profile[t]

        if t == 0:
            prob2 += soc_optimize[t] == initial_soc
        else:
            prob2 += soc_optimize[t] == soc_optimize[t-1] + (pv2batt2[t] + grid2batt2[t]) * charge_eff - batt2grid2[t] / discharge_eff

        prob2 += pv2batt2[t] + grid2batt2[t] <= charge_power_limit
        prob2 += batt2grid2[t] <= discharge_power_limit
        prob2 += pv2batt2[t] + grid2batt2[t] <= charge_power_limit * (1 - mode[t])
        prob2 += batt2grid2[t] <= discharge_power_limit * mode[t]
        prob2 += grid2batt2[t] <= grid_power_limit
        prob2 += pv2grid2[t] + batt2grid2[t] <= grid_power_limit

        if t == 0:
            prob2 += batt2grid2[t] <= initial_soc * discharge_eff
        else:
            prob2 += batt2grid2[t] <= soc_optimize[t-1] * discharge_eff
        prob2 += soc_optimize[T-1] == initial_soc

    # --- Target function ---
    profit_optimize = lpSum([
        (pv2grid2[t] + batt2grid2[t]) * (da_prices[t]-marginal_export) -
        grid2batt2[t] * (da_prices[t]*(1+VAT_tax)+marginal_import+consumption_tax+transmission_cost) -
        batt2grid2[t] * cycle_cost
        for t in range(T)
    ])

    prob2 += profit_optimize
    prob2.solve(pulp.PULP_CBC_CMD(msg=False))


    # --- Save the result ---
    profit_optimize_val = value(prob2.objective)
    results.append({
        "Battery Capacity (MWh)": cap,
        "Profit (€)": round(profit_optimize_val, 2),
    })


# STEP 5: PRINT THE RESULT & VISUALIZATION ---

##Print the power flow result
print("\n--- Optimized result---")
for t in range(T):
    hour_label = datetime(selected_date.year, selected_date.month, selected_date.day, t).strftime('%Y-%m-%d %H:%M')
    print(
        f"{hour_label} | PV: {pv_profile[t]:.2f} MW | Price: {da_prices[t]:.2f} €/MWh | "
        f"PV→BATT: {pv2batt[t].varValue:.2f} | PV→GRID: {pv2grid[t].varValue:.2f} | CURT: {curtail[t].varValue:.2f} | "
        f"GRID→BATT: {grid2batt[t].varValue:.2f} | BATT→GRID: {batt2grid[t].varValue:.2f} | SoC: {soc[t].varValue:.2f}"
    )

print(f"Status: {LpStatus[prob.status]}")
print(f"Objective (Profit): {value(prob.objective):.2f} €")
profit_value = value(prob.objective)


# --- Find the optimal battery capacity ---
df_results = pd.DataFrame(results)
best_row = df_results.loc[df_results["Profit (€)"].idxmax()]
print("\n📈 Profit by battery capacity:")
print(df_results)

print("\n🏆 Best configuration:")
print(best_row)

## Create the Excel file


time_labels = [datetime(selected_date.year, selected_date.month, selected_date.day, t).strftime('%Y-%m-%d %H:%M') for t in range(T)]
time_labels_short = [t[-5:] for t in time_labels] 


pv2batt_val = [pv2batt[t].varValue for t in range(T)]
pv2grid_val = [pv2grid[t].varValue for t in range(T)]
curtail_val = [curtail[t].varValue for t in range(T)]
grid2batt_val = [grid2batt[t].varValue for t in range(T)]
batt2grid_val = [batt2grid[t].varValue for t in range(T)]
soc_val = [value(soc[t])/battery_capacity*100 for t in range(T)]

df_data = pd.DataFrame({
    'Time': time_labels,
    'Day-Ahead Price (€/MWh)': da_prices,
    'Solar Irradiance (W/m²)': solar_irradiance,
    'PV Power (MW)': pv_profile,
    'PV → Grid (MW)': pv2grid_val,
    'PV → Battery (MW)': pv2batt_val,
    'PV Curtailment (MW)': curtail_val,
    'Grid → Battery (MW)': grid2batt_val,
    'Battery → Grid (MW)': batt2grid_val,
    'Battery SoC (%)': soc_val
})

df_config = pd.DataFrame({
    'Parameter': [
        'Objective (Profit) (€)',
        'PV solar rated power (MW)',
        'Grid rated power (MW)',
        'Battery maximum capacity (MWh)',
        'Battery discharge rated power (MW)',
        'Battery charge rated power (MW)',
        'Charge/Discharge efficiency (%)'
    ],
    'Value': [
        round(profit_value, 2),
        PV_rated,
        grid_power_limit,
        battery_capacity,
        discharge_power_limit,
        charge_power_limit,
        discharge_eff
        
    ]
})

output_path = f"/your path/energy_arbitrage_{selected_date.strftime('%Y-%m-%d')}.xlsx"
try:
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_config.to_excel(writer, sheet_name='Sheet1', index=False, startrow=0)
        start_row = len(df_config) + 3
        df_data.to_excel(writer, sheet_name='Sheet1', index=False, startrow=start_row)
        df_results.to_excel(writer, sheet_name='Sheet1', index=False, startrow=start_row + 27)

    print(f"✅ Excel file saved at:\n{output_path}")
except Exception as e:
    print(f"❌ Error when writing Excel file: {e}")

## Plot

fig = plt.figure(figsize=(12, 9))
fig.suptitle(f"Energy Arbitrage ({selected_date.strftime('%Y-%m-%d')})", fontsize=16)

# --- Subplot 1: PV profile + DA price ---
x = np.arange(len(time_labels))
ax1 = plt.subplot(4, 1, 1)

# Stacked bars
ax1.bar(x, pv2grid_val, label='PV → Grid', color='gold')
ax1.bar(x, pv2batt_val, bottom=pv2grid_val, label='PV → Battery', color='orange')
ax1.bar(x, curtail_val, bottom=np.array(pv2grid_val) + np.array(pv2batt_val),
        label='Curtailment', color='gray', alpha=0.7)

ax1.set_ylabel('Power (MW)')
ax1.set_title('PV Profile per Hour + DA Price')
ax1.set_xticks(x)
ax1.set_xticklabels(time_labels_short, rotation=45)
ax1.grid(True, axis='y')

# DA price (gray dashed)
ax1_price = ax1.twinx()
ax1_price.plot(x, da_prices, color='gray', linestyle='--', label='Day-Ahead Price (€/MWh)')
ax1_price.set_ylabel('€/MWh', color='gray')
ax1_price.tick_params(axis='y', labelcolor='gray')

# Legend
bars_labels = [ax1.patches[0], ax1.patches[len(x)], ax1.patches[2*len(x)]]
bar_names = ['PV → Grid', 'PV → Battery', 'Curtailment']
lines_labels = [ax1_price.lines[0]]
line_names = ['Day-Ahead Price (€/MWh)']
ax1.legend(bars_labels + lines_labels, bar_names + line_names, loc='upper left')

# --- Subplot 2: Battery SoC + DA price ---
ax2 = plt.subplot(4, 1, 2)
ax_soc = ax2
ax_price = ax2.twinx()

bar_width = 0.4
ax_soc.bar(x, [100]*len(soc_val), color='lightgray', width=bar_width, label='Max Capacity')
soc_bar = ax_soc.bar(x, soc_val, color='purple', width=bar_width, label='Battery SoC (%)')
ax_soc.set_ylabel('State of Charge (%)', color='purple')
ax_soc.set_ylim(0, 110)
ax_soc.tick_params(axis='y', labelcolor='purple')
ax_soc.set_title('Battery SoC and Day-Ahead Price')
ax_soc.set_xticks(x)
ax_soc.set_xticklabels(time_labels_short, rotation=45)
ax_soc.grid(True, axis='y')

price_line = ax_price.plot(x, da_prices, color='gray', linestyle='--', label='Day-Ahead Price (€/MWh)')
ax_price.set_ylabel('€/MWh', color='gray')
ax_price.tick_params(axis='y', labelcolor='gray')

ax_soc.legend([soc_bar[0], price_line[0]], ['Battery SoC (%)', 'Day-Ahead Price (€/MWh)'], loc='upper left')

# --- Subplot 3: Battery/Grid flows vs price ---
ax3 = plt.subplot(4, 1, 3)
ax_flow = ax3
ax_price2 = ax3.twinx()

flow_b2g = ax_flow.plot(x, batt2grid_val, color='blue', marker='s', label='Battery → Grid (MW)')
flow_g2b = ax_flow.plot(x, grid2batt_val, color='green', marker='^', label='Grid → Battery (MW)')
ax_flow.set_ylabel('Power (MW)')
ax_flow.set_xticks(x)
ax_flow.set_xticklabels(time_labels_short, rotation=45)
ax_flow.grid(True, axis='y')

price_line2 = ax_price2.plot(x, da_prices, color='gray', linestyle='--', label='Day-Ahead Price (€/MWh)')
ax_price2.set_ylabel('€/MWh', color='gray')

lines_labels3 = flow_b2g + flow_g2b + price_line2
labels3 = ['Battery → Grid (MW)', 'Grid → Battery (MW)', 'Day-Ahead Price (€/MWh)']
ax_flow.legend(lines_labels3, labels3, loc='upper left')
ax_flow.set_title('Battery/Grid Flows vs Day-Ahead Price')

# --- Subplot 4: Profit vs Battery Capacity ---
ax4 = plt.subplot(4, 1, 4)
ax4.plot(df_results["Battery Capacity (MWh)"], df_results["Profit (€)"], marker='o')
ax4.set_title("Profit vs Battery Capacity")
ax4.set_xlabel("Battery Capacity (MWh)")
ax4.set_ylabel("Profit (€)")
ax4.grid(True)
ax4.axvline(best_row["Battery Capacity (MWh)"], color='red', linestyle='--',
            label=f"Max Profit at {int(best_row['Battery Capacity (MWh)'])} MWh")
ax4.legend()

# --- Add system configuration (align with chart start) ---
system_info = (
    f"Profit (€): {profit_value:.2f}\n"
    f"PV solar rated power (MW): {PV_rated}\n"
    f"Grid rated power (MW): {grid_power_limit}\n"
    f"Battery maximum capacity (MWh): {battery_capacity}\n"
    f"Battery discharge rated power (MW): {discharge_power_limit}\n"
    f"Charge/Discharge efficiency: {discharge_eff}"
)
bbox = ax1.get_position()
plt.figtext(bbox.x0, 0.98, system_info, fontsize=7, ha='left', va='top')

# --- Layout adjustments ---
plt.tight_layout()                # Auto alignment
plt.subplots_adjust(top=0.9,     # Leave top space for suptitle
                    bottom=0.05)  # Leave bottom space for chart 4
plt.show()
