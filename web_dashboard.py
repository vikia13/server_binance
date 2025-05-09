import os
import json
import sqlite3
import time
import threading
from datetime import datetime
from flask import Flask, render_template, jsonify, request, redirect, url_for

app = Flask(__name__, template_folder='templates', static_folder='static')

# Global variables
trading_system = None
db_path = 'data'


def format_timestamp(ts):
    """Convert timestamp to readable date"""
    if not ts:
        return "N/A"
    return datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M:%S')


def format_pct(value):
    """Format percentage value"""
    if value is None:
        return "N/A"
    return f"{value:.2f}%"


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html',
                           system_status=trading_system.running if trading_system else False)


@app.route('/api/status')
def system_status():
    """Get system status"""
    if not trading_system:
        return jsonify({"status": "not_initialized"})

    return jsonify({
        "status": "running" if trading_system.running else "stopped",
        "symbols": trading_system.symbols,
        "uptime": time.time() - trading_system.start_time if hasattr(trading_system, 'start_time') else 0,
        "signal_count": trading_system.signal_generator.signal_counter
    })


@app.route('/api/active_signals')
def active_signals():
    """Get all active signals"""
    if not trading_system or not trading_system.signal_generator:
        return jsonify([])

    signals = trading_system.signal_generator.get_active_signals()

    # Format timestamps
    for signal in signals:
        signal['formatted_time'] = format_timestamp(signal['timestamp'])

    return jsonify(signals)


@app.route('/api/signals_history')
def signals_history():
    """Get historical signals"""
    if not trading_system or not trading_system.signal_generator:
        return jsonify([])

    limit = request.args.get('limit', 100, type=int)
    signals = trading_system.signal_generator.get_signals_history(limit=limit)

    # Format timestamps and percentages
    for signal in signals:
        signal['formatted_time'] = format_timestamp(signal['timestamp'])
        signal['formatted_closed_time'] = format_timestamp(signal['closed_at'])
        signal['formatted_profit'] = format_pct(signal['profit_pct'])

    return jsonify(signals)


@app.route('/api/performance')
def performance():
    """Get performance statistics"""
    conn = sqlite3.connect(os.path.join(db_path, 'signals.db'))
    cursor = conn.cursor()

    # Overall stats
    cursor.execute('''
    SELECT
        COUNT(*) as total_signals,
        SUM(CASE WHEN status = 'WIN' THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN status = 'LOSS' THEN 1 ELSE 0 END) as losses,
        AVG(CASE WHEN status = 'WIN' THEN profit_pct ELSE NULL END) as avg_win,
        AVG(CASE WHEN status = 'LOSS' THEN profit_pct ELSE NULL END) as avg_loss,
        SUM(profit_pct) as total_profit
    FROM signals
    WHERE status IN ('WIN', 'LOSS')
    ''')

    overall = cursor.fetchone()

    # Stats by symbol
    cursor.execute('''
    SELECT
        symbol,
        COUNT(*) as total,
        SUM(CASE WHEN status = 'WIN' THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN status = 'LOSS' THEN 1 ELSE 0 END) as losses,
        AVG(CASE WHEN status = 'WIN' THEN profit_pct ELSE NULL END) as avg_win,
        AVG(CASE WHEN status = 'LOSS' THEN profit_pct ELSE NULL END) as avg_loss,
        SUM(profit_pct) as total_profit
    FROM signals
    WHERE status IN ('WIN', 'LOSS')
    GROUP BY symbol
    ORDER BY total_profit DESC
    ''')

    symbols = cursor.fetchall()
    conn.close()

    return jsonify({
        "overall": {
            "total_signals": overall[0] or 0,
            "wins": overall[1] or 0,
            "losses": overall[2] or 0,
            "win_rate": (overall[1] / overall[0]) * 100 if overall[0] > 0 else 0,
            "avg_win": overall[3] or 0,
            "avg_loss": overall[4] or 0,
            "total_profit": overall[5] or 0
        },
        "symbols": [
            {
                "symbol": symbol[0],
                "total": symbol[1],
                "wins": symbol[2],
                "losses": symbol[3],
                "win_rate": (symbol[2] / symbol[1]) * 100 if symbol[1] > 0 else 0,
                "avg_win": symbol[4] or 0,
                "avg_loss": symbol[5] or 0,
                "total_profit": symbol[6] or 0
            }
            for symbol in symbols
        ]
    })


@app.route('/api/control', methods=['POST'])
def control_system():
    """Control the trading system"""
    if not trading_system:
        return jsonify({"success": False, "message": "Trading system not initialized"})

    action = request.json.get('action')

    if action == 'start' and not trading_system.running:
        threading.Thread(target=trading_system.start).start()
        return jsonify({"success": True, "message": "System starting..."})

    elif action == 'stop' and trading_system.running:
        trading_system.stop()
        return jsonify({"success": True, "message": "System stopped"})

    return jsonify({"success": False, "message": "Invalid action or system already in requested state"})


def create_templates():
    """Create HTML templates if they don't exist"""
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    # Create index.html
    index_html = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Trading System</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            .card { margin-bottom: 20px; }
            .signal-active { background-color: #d4edda; }
            .signal-win { background-color: #d1e7dd; }
            .signal-loss { background-color: #f8d7da; }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-dark bg-dark mb-4">
            <div class="container">
                <a class="navbar-brand" href="#">AI Trading System Dashboard</a>
                <div>
                    <button id="startBtn" class="btn btn-success">Start System</button>
                    <button id="stopBtn" class="btn btn-danger">Stop System</button>
                </div>
            </div>
        </nav>

        <div class="container">
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            System Status
                        </div>
                        <div class="card-body">
                            <p><strong>Status:</strong> <span id="systemStatus">Loading...</span></p>
                            <p><strong>Symbols:</strong> <span id="systemSymbols">Loading...</span></p>
                            <p><strong>Uptime:</strong> <span id="systemUptime">Loading...</span></p>
                            <p><strong>Signal Count:</strong> <span id="signalCount">Loading...</span></p>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            Performance Summary
                        </div>
                        <div class="card-body">
                            <canvas id="performanceChart"></canvas>
                            <div id="performanceDetails" class="mt-3">
                                <p><strong>Total Signals:</strong> <span id="totalSignals">Loading...</span></p>
                                <p><strong>Win Rate:</strong> <span id="winRate">Loading...</span></p>
                                <p><strong>Total Profit:</strong> <span id="totalProfit">Loading...</span></p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    Active Signals
                </div>
                <div class="card-body">
                    <div class="table-responsive">
                        <table class="table table-striped table-hover">
                            <thead>
                                <tr>
                                    <th>Symbol</th>
                                    <th>Direction</th>
                                    <th>Entry Price</th>
                                    <th>Stop Loss</th>
                                    <th>Take Profit</th>
                                    <th>Risk/Reward</th>
                                    <th>Confidence</th>
                                    <th>Status</th>
                                    <th>Time</th>
                                </tr>
                            </thead>
                            <tbody id="activeSignalsBody">
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    Signal History
                </div>
                <div class="card-body">
                    <div class="table-responsive">
                        <table class="table table-striped table-hover">
                            <thead>
                                <tr>
                                    <th>Symbol</th>
                                    <th>Direction</th>
                                    <th>Entry Price</th>
                                    <th>Status</th>
                                    <th>Profit %</th>
                                    <th>Time</th>
                                    <th>Closed At</th>
                                </tr>
                            </thead>
                            <tbody id="signalHistoryBody">
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Update system status
            function updateSystemStatus() {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('systemStatus').textContent = data.status;
                        document.getElementById('systemSymbols').textContent = data.symbols.join(', ');

                        const hours = Math.floor(data.uptime / 3600);
                        const minutes = Math.floor((data.uptime % 3600) / 60);
                        document.getElementById('systemUptime').textContent = `${hours}h ${minutes}m`;

                        document.getElementById('signalCount').textContent = data.signal_count;

                        // Update button states
                        document.getElementById('startBtn').disabled = (data.status === 'running');
                        document.getElementById('stopBtn').disabled = (data.status !== 'running');
                    });
            }

            // Update active signals
            function updateActiveSignals() {
                fetch('/api/active_signals')
                    .then(response => response.json())
                    .then(signals => {
                        const tableBody = document.getElementById('activeSignalsBody');
                        tableBody.innerHTML = '';

                        if (signals.length === 0) {
                            tableBody.innerHTML = '<tr><td colspan="9" class="text-center">No active signals</td></tr>';
                            return;
                        }

                        signals.forEach(signal => {
                            const row = document.createElement('tr');
                            row.className = 'signal-active';

                            row.innerHTML = `
                                <td>${signal.symbol}</td>
                                <td>${signal.direction}</td>
                                <td>${signal.entry_price}</td>
                                <td>${signal.stop_loss}</td>
                                <td>${signal.take_profit}</td>
                                <td>${signal.risk_reward_ratio.toFixed(2)}</td>
                                <td>${(signal.confidence * 100).toFixed(1)}%</td>
                                <td>${signal.status}</td>
                                <td>${signal.formatted_time}</td>
                            `;

                            tableBody.appendChild(row);
                        });
                    });
            }

            // Update signal history
            function updateSignalHistory() {
                fetch('/api/signals_history')
                    .then(response => response.json())
                    .then(signals => {
                        const tableBody = document.getElementById('signalHistoryBody');
                        tableBody.innerHTML = '';

                        if (signals.length === 0) {
                            tableBody.innerHTML = '<tr><td colspan="7" class="text-center">No signal history</td></tr>';
                            return;
                        }

                        signals.forEach(signal => {
                            const row = document.createElement('tr');
                            if (signal.status === 'WIN') {
                                row.className = 'signal-win';
                            } else if (signal.status === 'LOSS') {
                                row.className = 'signal-loss';
                            }

                            row.innerHTML = `
                                <td>${signal.symbol}</td>
                                <td>${signal.direction}</td>
                                <td>${signal.entry_price}</td>
                                <td>${signal.status}</td>
                                <td>${signal.formatted_profit}</td>
                                <td>${signal.formatted_time}</td>
                                <td>${signal.formatted_closed_time}</td>
                            `;

                            tableBody.appendChild(row);
                        });
                    });
            }

            // Update performance statistics
            function updatePerformance() {
                fetch('/api/performance')
                    .then(response => response.json())
                    .then(data => {
                        // Update summary stats
                        document.getElementById('totalSignals').textContent = data.overall.total_signals;
                        document.getElementById('winRate').textContent = data.overall.win_rate.toFixed(1) + '%';
                        document.getElementById('totalProfit').textContent = data.overall.total_profit.toFixed(2) + '%';

                        // Create performance chart
                        const symbols = data.symbols.slice(0, 5).map(s => s.symbol);
                        const profits = data.symbols.slice(0, 5).map(s => s.total_profit);

                        if (window.perfChart) {
                            window.perfChart.destroy();
                        }

                        const ctx = document.getElementById('performanceChart').getContext('2d');
                        window.perfChart = new Chart(ctx, {
                            type: 'bar',
                            data: {
                                labels: symbols,
                                datasets: [{
                                    label: 'Total Profit (%)',
                                    data: profits,
                                    backgroundColor: profits.map(p => p >= 0 ? 'rgba(75, 192, 192, 0.6)' : 'rgba(255, 99, 132, 0.6)'),
                                    borderColor: profits.map(p => p >= 0 ? 'rgba(75, 192, 192, 1)' : 'rgba(255, 99, 132, 1)'),
                                    borderWidth: 1
                                }]
                            },
                            options: {
                                scales: {
                                    y: {
                                        beginAtZero: false
                                    }
                                },
                                plugins: {
                                    legend: {
                                        display: false
                                    },
                                    title: {
                                        display: true,
                                        text: 'Top 5 Symbols by Profit'
                                    }
                                }
                            }
                        });
                    });
            }

            // Control system
            document.getElementById('startBtn').addEventListener('click', function() {
                fetch('/api/control', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ action: 'start' }),
                })
                .then(response => response.json())
                .then(data => {
                    alert(data.message);
                    setTimeout(updateSystemStatus, 1000);
                });
            });

            document.getElementById('stopBtn').addEventListener('click', function() {
                fetch('/api/control', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ action: 'stop' }),
                })
                .then(response => response.json())
                .then(data => {
                    alert(data.message);
                    setTimeout(updateSystemStatus, 1000);
                });
            });

            // Initial update
            updateSystemStatus();
            updateActiveSignals();
            updateSignalHistory();
            updatePerformance();

            // Set update intervals
            setInterval(updateSystemStatus, 5000);
            setInterval(updateActiveSignals, 10000);
            setInterval(updateSignalHistory, 15000);
            setInterval(updatePerformance, 30000);
        </script>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    '''

    with open('templates/index.html', 'w') as f:
        f.write(index_html)


def run_dashboard(system, port=5000, host='0.0.0.0'):
    """Start the web dashboard"""
    global trading_system, db_path
    trading_system = system
    db_path = system.db_path

    # Create templates
    create_templates()

    # Store start time for uptime calculation
    if trading_system and not hasattr(trading_system, 'start_time'):
        trading_system.start_time = time.time()

    # Run Flask app
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    print("This module should be imported and run with an AI Trading System instance.")
    print("Example: run_dashboard(trading_system, port=5000)")