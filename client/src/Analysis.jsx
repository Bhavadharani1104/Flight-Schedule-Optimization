import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import api from "./api";
import './analysis.css';

function Analysis() {
  const [flights, setFlights] = useState([]);
  const [heatmap, setHeatmap] = useState({ z: [], x: [], y: [], text: [] });
  const [slotData, setSlotData] = useState({ x: [], y: [] });
  const [impactData, setImpactData] = useState({ x: [], y: [] });

  useEffect(() => {
    api.get("/api/flights").then(({ data }) => {
      setFlights(data);

      // ---------------- Heatmap ----------------
      const stdTimes = Array.from(new Set(
        data.map(f => f.STD ? String(f.STD).slice(0,5) : null).filter(Boolean)
      )).sort();

      const runways = Array.from(new Set(data.map(f => f.Runway).filter(Boolean))).sort();

      const zMatrix = [];
      const textMatrix = [];
      for (const rwy of runways) {
        const zRow = [];
        const textRow = [];
        for (const std of stdTimes) {
          const flightsHere = data.filter(f => f.Runway === rwy && f.STD && String(f.STD).slice(0,5) === std);
          const totalDelay = flightsHere.reduce((sum, f) =>
            sum + (f.TotalPredictedDelay && !isNaN(Number(f.TotalPredictedDelay)) ? Number(f.TotalPredictedDelay) : 0), 0
          );
          zRow.push(totalDelay);
          textRow.push(flightsHere.length > 0 ? 
            `Total Delay: ${totalDelay.toFixed(2)} min<br>` +
            flightsHere.map(f => `${f["Flight Number"] || ""}: ${Number(f.TotalPredictedDelay).toFixed(2)} min`).join("<br>")
            : ""
          );
        }
        zMatrix.push(zRow);
        textMatrix.push(textRow);
      }
      setHeatmap({ z: zMatrix, x: stdTimes, y: runways, text: textMatrix });

      // ---------------- Busiest slots ----------------
      const slotCounts = {};
      data.forEach(f => { 
        if(f.STD){ 
          const hour = String(f.STD).slice(0,2); 
          slotCounts[hour] = (slotCounts[hour]||0)+1; 
        }
      });
      const slotX = Object.keys(slotCounts).sort();
      const slotY = slotX.map(h => slotCounts[h]);
      setSlotData({ x: slotX, y: slotY });

      // ---------------- High-Impact flights ----------------
      const sorted = data.filter(f => f.TotalPredictedDelay != null)
                         .sort((a,b)=> Number(b.TotalPredictedDelay)-Number(a.TotalPredictedDelay))
                         .slice(0,10);
      setImpactData({ x: sorted.map(f=>f["Flight Number"]), y: sorted.map(f=>Number(f.TotalPredictedDelay)) });
    });
  }, []);

  return (
    <div className="analysis-container">
      <h1 className="analysis-main-title">📊 Flight Delay Analysis</h1>

      {/* Total Delay Heatmap */}
      <div className="chart-section">
        <h2 className="chart-subtitle">1️⃣ Total Predicted Delay Heatmap</h2>
        <p className="chart-description">
          X-axis: Scheduled Departure Time (HH:MM) <br/>
          Y-axis: Runway <br/>
          Color intensity: Total predicted delay (minutes)
        </p>
        <Plot
          data={[{
            z: heatmap.z, x: heatmap.x, y: heatmap.y,
            type: "heatmap", colorscale: "Viridis", text: heatmap.text,
            hoverinfo: "text+z", showscale: true, colorbar: { title: "Total Delay (min)" }
          }]}
          layout={{
            width: 1100, height: 520, margin:{t:60,l:80,r:40,b:120},
            xaxis:{title:"STD (HH:MM)", tickangle:-45, automargin:true},
            yaxis:{title:"Runway", automargin:true}
          }}
          config={{ responsive:true }}
        />
      </div>

      {/* Busiest Departure Hours */}
      <div className="chart-section">
        <h2 className="chart-subtitle">2️⃣ Busiest Departure Hours</h2>
        <p className="chart-description">
          X-axis: Hour of the day (0–23) <br/>
          Y-axis: Number of flights scheduled
        </p>
        <Plot
          data={[{ x: slotData.x, y: slotData.y, type:"bar", marker:{color:"skyblue"} }]}
          layout={{
            width:700, height:350,
            xaxis:{title:"Hour (24h)"},
            yaxis:{title:"Flights Count"},
            margin:{t:60,l:60,r:40,b:60}
          }}
          config={{ responsive:true }}
        />
      </div>

      {/* Top 10 High-Impact Flights */}
      <div className="chart-section">
        <h2 className="chart-subtitle">3️⃣ Top 10 High-Impact Flights</h2>
        <p className="chart-description">
          X-axis: Total Predicted Delay (minutes) <br/>
          Y-axis: Flight Number
        </p>
        <Plot
          data={[{
            x: impactData.y.slice().reverse(),
            y: impactData.x.slice().reverse(),
            type:"bar",
            orientation:"h",
            marker:{color:"salmon"}
          }]}
          layout={{
            width:700, height:400,
            xaxis:{title:"Total Predicted Delay (min)"},
            yaxis:{title:"Flight Number"},
            margin:{t:60,l:120,r:40,b:60}
          }}
          config={{ responsive:true }}
        />
      </div>
    </div>
  );
}

export default Analysis;
