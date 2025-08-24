import React, { useEffect, useRef, useState } from "react";
import { io } from "socket.io-client";
import { Paper, Typography, TextField, Button } from "@mui/material";
import './chatbot.css';

const socket = io("http://127.0.0.1:5000");

function Chatbot() {
  const [msgs, setMsgs] = useState([{ sender:"bot", text:"Hi! Ask me about flights, delays, and runways. Type 'help' for options." }]);
  const [input, setInput] = useState("");
  const endRef = useRef(null);

  useEffect(() => {
    socket.on("chat_reply", ({ text }) => setMsgs(prev => [...prev, { sender:"bot", text }]));
    return () => socket.off("chat_reply");
  }, []);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [msgs]);

  const send = () => {
    if(!input.trim()) return;
    const text = input.trim();
    setMsgs(prev => [...prev, { sender:"user", text }]);
    socket.emit("chat_message", { text });
    setInput("");
  };

  return (
    <div className="chatbot-container">
      <Typography variant="h5" className="chatbot-title">Assistant</Typography>
      <Paper className="chatbox" elevation={3}>
        {msgs.map((m,i) => (
          <div key={i} className={`message-wrapper ${m.sender}`}>
            <div className={`message-bubble ${m.sender}`}>{m.text}</div>
          </div>
        ))}
        <div ref={endRef} />
      </Paper>
      <div className="chat-input-wrapper">
        <TextField
          className="chat-input"
          fullWidth
          placeholder="e.g., Predicted delay for AI2509"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key==="Enter" && send()}
        />
        <Button className="chat-send-btn" variant="contained" onClick={send}>Send</Button>
      </div>
    </div>
  );
}

export default Chatbot;
