let active = false, loop = null;

// 1. Initialize Charts
const createChart = (id, label, color, fill = false) => {
    const ctx = document.getElementById(id).getContext('2d');
    return new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: [{ label: label, data: [], borderColor: color, backgroundColor: color + '22', fill: fill, tension: 0.3, pointRadius: 0 }] },
        options: { 
            responsive: true, maintainAspectRatio: false, 
            plugins: { legend: { display: false } },
            scales: { x: { display: false }, y: { grid: { color: '#1e293b' }, ticks: { color: '#64748b', font: { size: 9 } } } } 
        }
    });
};

const rewardChart = createChart('rewardChart', 'Reward', '#3b82f6', true);
const revenueChart = createChart('revenueChart', 'Revenue', '#10b981');
const ctrChart = createChart('ctrChart', 'CTR', '#a855f7');

// 2. Start/Stop Logic
document.getElementById('startBtn').onclick = () => {
    active = !active;
    document.getElementById('startBtn').innerText = active ? 'STOP_AGENT' : 'START_TRAINING';
    document.getElementById('startBtn').className = active ? 'w-full bg-red-700 py-2 rounded font-bold text-xs' : 'w-full bg-blue-700 py-2 rounded font-bold text-xs';
    
    if (active) {
        loop = setInterval(async () => {
            try {
                const res = await fetch('/train_step', { method: 'POST' });
                const data = await res.json();
                updateUI(data);
            } catch (e) { console.error("Update Error:", e); }
        }, 400);
    } else {
        clearInterval(loop);
    }
};

// 3. UI Update Engine
function updateUI(d) {
    // Stats Update
    document.getElementById('revenue').innerText = `₹${d.stats.revenue.toLocaleString('en-IN')}`;
    document.getElementById('roi').innerText = `${((d.stats.revenue/d.stats.spend)*100 || 0).toFixed(1)}%`;
    document.getElementById('ctr').innerText = `${d.stats.ctr}%`;
    document.getElementById('spend').innerText = `₹${d.stats.spend.toLocaleString('en-IN')}`;
    document.getElementById('epsilon').innerText = d.epsilon;

    // Intent Bar
    const iVal = Math.round(d.intent * 100);
    document.getElementById('intentText').innerText = iVal + "%";
    document.getElementById('intentBar').style.width = iVal + "%";

    // Chart Data Handling
    const updateGraph = (chart, newVal) => {
        chart.data.labels.push("");
        chart.data.datasets[0].data.push(newVal);
        if (chart.data.labels.length > 50) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }
        chart.update('none');
    };

    updateGraph(rewardChart, d.reward);
    updateGraph(revenueChart, d.stats.revenue);
    updateGraph(ctrChart, d.stats.ctr);

    // Live Feed Logs
    const log = document.createElement('div');
    log.className = `p-1 border-b border-slate-900 ${d.reward > 0 ? 'text-green-400' : 'text-slate-600'}`;
    log.innerHTML = `<span class="text-blue-500">BID: ₹${d.action}</span> | <span class="text-slate-400">REW: ${d.reward}</span>`;
    const box = document.getElementById('logBox');
    box.prepend(log);
    
    // Safety: prevent browser from slowing down by limiting DOM elements
    if (box.children.length > 100) box.removeChild(box.lastChild);
}

// Save/Load
document.getElementById('saveBtn').onclick = () => fetch('/save', {method:'POST'}).then(() => alert("Saved"));
document.getElementById('loadBtn').onclick = () => fetch('/load', {method:'POST'}).then(() => alert("Loaded"));