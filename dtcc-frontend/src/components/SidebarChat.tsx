import React, { useState, useRef, useEffect } from "react";

type Message = {
  role: "user" | "rag";
  content: string;
};

const SidebarChat: React.FC = () => {
  const [open, setOpen] = useState(false);
  const [userInput, setUserInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (open && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, open]);

  const sendMessage = async () => {
    if (!userInput.trim()) return;
    setMessages((msgs) => [...msgs, { role: "user", content: userInput }]);
    setLoading(true);
    setError("");
    try {
      const res = await fetch("http://localhost:8080/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userInput }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      let reply = data.response || "No response from LLM.";
      setMessages((msgs) => [...msgs, { role: "rag", content: reply }]);
    } catch (e: any) {
      setError("Error: " + e.message);
    }
    setLoading(false);
    setUserInput("");
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!loading && userInput.trim()) sendMessage();
  };

  return (
    <>
      {/* Toggle Button */}
      <button
        className="fixed top-4 right-4 z-50 bg-blue-600 text-white rounded-full w-12 h-12 flex items-center justify-center shadow-lg hover:bg-blue-700 transition"
        onClick={() => setOpen((v) => !v)}
        aria-label="Toggle chat sidebar"
      >
        {open ? "×" : "💬"}
      </button>
      {/* Sidebar */}
      <div
        className={`fixed top-0 right-0 h-full w-80 bg-white shadow-lg border-l border-gray-200 z-40 transform transition-transform duration-300 flex flex-col ${
          open ? "translate-x-0" : "translate-x-full"
        }`}
      >
        <div className="flex items-center justify-between px-4 py-3 border-b bg-gray-50">
          <span className="font-semibold text-lg">💬 Ask about your spending</span>
          <button
            className="text-2xl text-gray-500 hover:text-gray-700"
            onClick={() => setOpen(false)}
            aria-label="Close sidebar"
          >
            ×
          </button>
        </div>
        <div className="flex-1 overflow-y-auto px-4 py-2 bg-gray-50">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`mb-3 ${
                msg.role === "user"
                  ? "text-right text-blue-700"
                  : "text-left text-gray-800 bg-blue-50 rounded px-3 py-2 inline-block"
              }`}
            >
              {msg.content}
            </div>
          ))}
          {loading && (
            <div className="text-left text-gray-500 bg-blue-50 rounded px-3 py-2 inline-block mb-3">
              Thinking...
            </div>
          )}
          {error && (
            <div className="text-left text-red-600 bg-red-50 rounded px-3 py-2 inline-block mb-3">
              {error}
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        <form
          className="flex items-center border-t px-3 py-2 bg-white"
          onSubmit={handleSubmit}
        >
          <input
            type="text"
            className="flex-1 rounded border border-gray-300 px-3 py-2 mr-2 focus:outline-none focus:ring-2 focus:ring-blue-200 bg-gray-50"
            placeholder="Ask about your expenditure..."
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            disabled={loading}
            autoComplete="off"
          />
          <button
            type="submit"
            className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
            disabled={loading || !userInput.trim()}
          >
            Send
          </button>
        </form>
      </div>
    </>
  );
};

export default SidebarChat;