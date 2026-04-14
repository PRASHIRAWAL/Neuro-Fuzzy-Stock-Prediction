let probChartInstance = null;
let gaugeChartInstance = null;

// dynamic slider text updates + gradient fill
['volatility', 'trend', 'volume'].forEach(id => {
    const slider = document.getElementById(id);
    const label = document.getElementById(`${id}-val`);

    // Add event listener
    slider.addEventListener('input', (e) => {
        label.textContent = e.target.value;
    });
});

let lastInputData = null;
let lastPredictionData = null;

document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    // UI states
    const btnText = document.querySelector('.btn-text');
    const btnLoader = document.getElementById('btn-loader');
    const errorMsg = document.getElementById('error-message');

    btnText.style.display = 'none';
    btnLoader.style.display = 'block';
    errorMsg.textContent = '';

    // Collect data
    const data = {
        volatility: parseFloat(document.getElementById('volatility').value),
        trend: parseFloat(document.getElementById('trend').value),
        volume: parseFloat(document.getElementById('volume').value)
    };
    lastInputData = data;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'Prediction failed');
        }

        lastPredictionData = result;
        displayResults(result);
    } catch (err) {
        errorMsg.textContent = err.message;
    } finally {
        btnText.style.display = 'block';
        btnLoader.style.display = 'none';
    }
});

function displayResults(data) {
    document.getElementById('result-placeholder').style.display = 'none';
    document.getElementById('results-content').style.display = 'block';

    // Update Badge
    const riskSpan = document.getElementById('risk-text');
    riskSpan.parentElement.className = 'risk-badge'; // reset
    riskSpan.parentElement.classList.add(`risk-${data.predicted_risk.toLowerCase()}`);
    riskSpan.textContent = data.predicted_risk + " RISK";

    // Animate Continuous Score Text
    animateValue('continuous-score', 0, data.defuzzified_value, 1500);

    // Update Explainability Bars smoothly
    document.getElementById('explain-message').innerHTML = data.explainability.message;

    setTimeout(() => {
        document.getElementById('bar-volatility').style.width = `${data.explainability.contributions.Volatility}%`;
        document.getElementById('bar-trend').style.width = `${data.explainability.contributions.Trend}%`;
        document.getElementById('bar-volume').style.width = `${data.explainability.contributions.Volume}%`;
    }, 100);

    // Render Charts
    updateGaugeChart(data.defuzzified_value, data.predicted_risk);
    updateProbabilityChart(data.probabilities);
}

function animateValue(id, start, end, duration) {
    let obj = document.getElementById(id);
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const ease = 1 - Math.pow(1 - progress, 4);
        obj.innerHTML = (progress * (end - start) + start).toFixed(1);
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

function getRiskColor(riskLevel) {
    if (riskLevel === 'Low') return '#10b981';
    if (riskLevel === 'Medium') return '#f59e0b';
    return '#ef4444';
}

function updateGaugeChart(score, riskLevel) {
    const ctx = document.getElementById('gaugeChart').getContext('2d');
    if (gaugeChartInstance) gaugeChartInstance.destroy();

    const remainder = 10 - score;
    const color = getRiskColor(riskLevel);

    gaugeChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [score, remainder],
                backgroundColor: [color, 'rgba(255,255,255,0.05)'],
                borderWidth: 0,
                borderRadius: [10, 0] // Only round the end
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            rotation: 270,
            circumference: 180,
            cutout: '80%',
            plugins: {
                tooltip: { enabled: false },
                legend: { display: false }
            },
            animation: {
                animateScale: true,
                animateRotate: true,
                duration: 1500,
                easing: 'easeOutBounce'
            }
        }
    });
}

function updateProbabilityChart(probs) {
    const ctx = document.getElementById('probabilityChart').getContext('2d');

    if (probChartInstance) {
        probChartInstance.destroy();
    }

    probChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Low', 'Medium', 'High'],
            datasets: [{
                label: 'NN Probability Distribution',
                data: [probs.Low, probs.Medium, probs.High],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(245, 158, 11, 0.8)',
                    'rgba(239, 68, 68, 0.8)'
                ],
                borderColor: ['transparent', 'transparent', 'transparent'],
                borderWidth: 0,
                borderRadius: 8,
                barPercentage: 0.6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    grid: { color: 'rgba(255, 255, 255, 0.05)', drawBorder: false },
                    ticks: { color: '#64748b', padding: 10 }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8', font: { size: 14, weight: '500' } }
                }
            },
            plugins: {
                legend: { display: false }
            },
            animation: {
                duration: 1200,
                easing: 'easeOutQuart'
            }
        }
    });
}

function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(el => {
        el.style.opacity = '0';
        setTimeout(() => el.style.display = 'none', 300);
    });
    document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));

    setTimeout(() => {
        const target = document.getElementById(tabId);
        target.style.display = 'flex';
        if (tabId === 'cmTab') target.style.flexDirection = 'column';
        target.offsetHeight;
        target.style.opacity = '1';
    }, 300);

    event.currentTarget.classList.add('active');
}

// PDF Export Logic: Clean Template Generation
const btnDownloadPdf = document.getElementById('btn-download-pdf');
if (btnDownloadPdf) {
    btnDownloadPdf.addEventListener('click', () => {
        if (!lastPredictionData || !lastInputData) {
            alert('Please generate a prediction first!');
            return;
        }

        btnDownloadPdf.innerHTML = 'Generating Clean Report...';
        btnDownloadPdf.disabled = true;

        const dateStr = new Date().toLocaleString();

        // Build the clean HTML string - Added explicit width and box-sizing to prevent canvas collapse
        const htmlTemplate = `
            <div style="font-family: Arial, sans-serif; padding: 40px; color: #1e293b; background: white; width: 800px; box-sizing: border-box; margin: 0;">
                <h1 style="text-align: center; color: #0f172a; margin-bottom: 5px;">Stock Market Risk Prediction Report</h1>
                <p style="text-align: center; color: #64748b; margin-top: 0; margin-bottom: 30px;">
                    Generated: ${dateStr}<br>
                    Neuro-Fuzzy AI System
                </p>
                <hr style="border: 1px solid #e2e8f0; margin-bottom: 30px;">
                
                <h2 style="color: #334155; border-bottom: 2px solid #3b82f6; padding-bottom: 5px; margin-bottom: 15px;">Input Parameters</h2>
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 30px;">
                    <tr style="background-color: #3b82f6; color: white;">
                        <th style="padding: 10px; text-align: left; border: 1px solid #cbd5e1;">Parameter</th>
                        <th style="padding: 10px; text-align: left; border: 1px solid #cbd5e1;">Value</th>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #cbd5e1;">Market Volatility</td>
                        <td style="padding: 10px; border: 1px solid #cbd5e1;">${lastInputData.volatility} / 250</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #cbd5e1;">Market Trend</td>
                        <td style="padding: 10px; border: 1px solid #cbd5e1;">${lastInputData.trend} (-50 to +50)</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #cbd5e1;">Trading Volume</td>
                        <td style="padding: 10px; border: 1px solid #cbd5e1;">${lastInputData.volume}</td>
                    </tr>
                </table>

                <h2 style="color: #334155; border-bottom: 2px solid #f59e0b; padding-bottom: 5px; margin-bottom: 15px;">Risk Assessment Results</h2>
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 30px;">
                    <tr style="background-color: #f59e0b; color: white;">
                        <th style="padding: 10px; text-align: left; border: 1px solid #cbd5e1;">Metric</th>
                        <th style="padding: 10px; text-align: left; border: 1px solid #cbd5e1;">Value</th>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #cbd5e1; font-weight: bold;">Final Risk Level</td>
                        <td style="padding: 10px; border: 1px solid #cbd5e1; font-weight: bold; color: ${getRiskColor(lastPredictionData.predicted_risk)}">${lastPredictionData.predicted_risk.toUpperCase()}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #cbd5e1;">Calculated Fuzzy Score</td>
                        <td style="padding: 10px; border: 1px solid #cbd5e1;">${lastPredictionData.defuzzified_value.toFixed(2)} / 10</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #cbd5e1;">Neural Network Top Class Probability</td>
                        <td style="padding: 10px; border: 1px solid #cbd5e1;">${(Math.max(lastPredictionData.probabilities.Low, lastPredictionData.probabilities.Medium, lastPredictionData.probabilities.High) * 100).toFixed(1)}%</td>
                    </tr>
                </table>

                <h2 style="color: #334155; border-bottom: 2px solid #10b981; padding-bottom: 5px; margin-bottom: 10px;">Explainability Analysis</h2>
                <p style="line-height: 1.6; color: #475569; margin-bottom: 30px;">
                    ${lastPredictionData.explainability.message}
                </p>
                
                <h2 style="color: #334155; border-bottom: 2px solid #8b5cf6; padding-bottom: 5px; margin-bottom: 15px;">Fuzzy Rules Inference Mapping</h2>
                <p style="line-height: 1.5; color: #475569; margin-bottom: 30px; background: #f8fafc; padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <strong>Rule 1:</strong> IF Neural Score is Low THEN Risk is Low<br>
                    <strong>Rule 2:</strong> IF Neural Score is Medium THEN Risk is Medium<br>
                    <strong>Rule 3:</strong> IF Neural Score is High THEN Risk is High
                </p>
                
                <div style="margin-top: 50px; text-align: center; color: #94a3b8; font-size: 0.85rem;">
                    <p>End of Report</p>
                </div>
            </div>
        `;

        // Ditch html2pdf.js entirely - it fails on complex DOM bindings.
        // We will use the absolutely flawless built-in native browser print engine (Vector PDF capability).

        const printContainer = document.createElement('div');
        printContainer.id = "native-pdf-print";
        printContainer.innerHTML = htmlTemplate;

        // Prepare global styles to hide the rest of the app during the instant PDF generation
        const printStyle = document.createElement('style');
        printStyle.innerHTML = `
            @media print {
                body > * {
                    visibility: hidden;
                    display: none !important;
                }
                body > #native-pdf-print {
                    visibility: visible !important;
                    display: block !important;
                    position: absolute;
                    left: 0;
                    top: 0;
                    width: 100%;
                    margin: 0;
                    background: white;
                }
                @page {
                    margin: 0.5in;
                    size: auto;
                }
            }
        `;

        document.head.appendChild(printStyle);
        document.body.appendChild(printContainer);

        // Allow tiny delay for CSS to mount, then execute native browser print vectorization
        setTimeout(() => {
            window.print();

            // Cleanup DOM so the app returns to normal immediately after the dialog
            document.head.removeChild(printStyle);
            document.body.removeChild(printContainer);

            btnDownloadPdf.innerHTML = '⬇ Download PDF Report';
            btnDownloadPdf.disabled = false;
        }, 100);
    });
}
