import React, { useEffect, useMemo, useState } from "react";
import api from "./api";
import { io } from "socket.io-client";
import { DataGrid } from "@mui/x-data-grid";
import { Paper, Typography } from "@mui/material";
import './dashboard.css';

const socket = io("http://127.0.0.1:5000");

function Dashboard() {
  const [rows, setRows] = useState([]);

  // Initial load
  useEffect(() => {
    api.get("/api/flights")
      .then(({ data }) => {
        const normalized = data.map((r, i) => ({
          id: r.rowId || i + 1,
          ...r
        }));
        setRows(normalized);
      })
      .catch(err => console.error("API fetch error:", err));
  }, []);

  // Live updates via socket
  useEffect(() => {
    socket.on("flight_updated", ({ row }) => {
      setRows(prev =>
        prev.map(r =>
          r.rowId === row.rowId
            ? { ...r, ...row, id: row.rowId }
            : r
        )
      );
    });
    return () => socket.off("flight_updated");
  }, []);

  const columns = useMemo(() => [
    { field: "Flight Number", headerName: "Flight", minWidth: 140 },
    { field: "From", headerName: "From", minWidth: 160 },
    { field: "To", headerName: "To", minWidth: 160 },
    { field: "Aircraft", headerName: "Aircraft", minWidth: 140 },
    { field: "STD", headerName: "STD", minWidth: 110, editable: true },
    { field: "Optimal_STD", headerName: "Optimal STD", minWidth: 130, editable: true },
    { field: "Predicted_Delay", headerName: "Predicted", minWidth: 110 },
    { field: "CascadingDelay", headerName: "Cascade", minWidth: 110 },
    { field: "TotalPredictedDelay", headerName: "Total", minWidth: 110 },
    { field: "Runway", headerName: "Runway", minWidth: 100 },
    { field: "FlightType", headerName: "Type", minWidth: 110 }
  ], []);

  const processRowUpdate = async (newRow, oldRow) => {
    let field = null;
    if (newRow.STD !== oldRow.STD) field = "STD";
    else if (newRow.Optimal_STD !== oldRow.Optimal_STD) field = "Optimal_STD";
    if (!field) return oldRow;

    const value = newRow[field];
    try {
      const { data } = await api.post("/api/update", {
        rowId: newRow.rowId || newRow.id,
        field,
        value
      });

      if (data.status === "success" && data.updatedRow) {
        return { ...newRow, ...data.updatedRow, id: data.updatedRow.rowId };
      } else {
        alert(data.message || "Update failed");
        return oldRow;
      }
    } catch (err) {
      console.error("Update error:", err);
      return oldRow;
    }
  };

  return (
    <div className="dashboard-container">
      <Typography className="dashboard-title">Flight Scheduler Dashboard</Typography>
      <Paper className="dashboard-grid-container">
        <DataGrid
          rows={rows}
          columns={columns}
          density="compact"
          experimentalFeatures={{ newEditingApi: true }}
          processRowUpdate={processRowUpdate}
          onProcessRowUpdateError={console.error}
          getRowId={r => r.rowId || r.id}
          pageSizeOptions={[25, 50, 100]}
          initialState={{
            pagination: { paginationModel: { pageSize: 25, page: 0 } },
            sorting: { sortModel: [{ field: "STD", sort: "asc" }] }
          }}
          className="data-grid"
        />
      </Paper>
    </div>
  );
}

export default Dashboard;
