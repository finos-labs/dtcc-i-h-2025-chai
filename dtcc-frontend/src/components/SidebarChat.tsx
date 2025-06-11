"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { Send, MessageSquare, X, GripVertical } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"

type Message = {
  role: "user" | "rag"
  content: string
}

const formatMessage = (content: string) => {
  return content.split("\n").map((line, index) => {
    // Handle bullet points
    if (line.trim().startsWith("•")) {
      return (
        <div key={index} className="flex items-start gap-2 my-1">
          <span className="text-primary mt-0.5">•</span>
          <span>{line.trim().substring(1).trim()}</span>
        </div>
      )
    }

    // Handle empty lines for spacing
    if (line.trim() === "") {
      return <div key={index} className="h-2" />
    }

    // Regular lines
    return (
      <div key={index} className="my-1">
        {line}
      </div>
    )
  })
}

const SidebarChat: React.FC = () => {
  const [open, setOpen] = useState(false)
  const [userInput, setUserInput] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [messages, setMessages] = useState<Message[]>([])
  const [width, setWidth] = useState(320) // Default width
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const sidebarRef = useRef<HTMLDivElement>(null)
  const resizingRef = useRef(false)
  const startXRef = useRef(0)
  const startWidthRef = useRef(0)

  useEffect(() => {
    if (open && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" })
    }
  }, [messages, open])

  const sendMessage = async () => {
    if (!userInput.trim()) return
    setMessages((msgs) => [...msgs, { role: "user", content: userInput }])
    setLoading(true)
    setError("")
    try {
      const res = await fetch("http://localhost:8080/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userInput }),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      const reply = data.response || "No response from LLM."
      setMessages((msgs) => [...msgs, { role: "rag", content: reply }])
    } catch (e: any) {
      setError("Error: " + e.message)
    }
    setLoading(false)
    setUserInput("")
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!loading && userInput.trim()) sendMessage()
  }

  // Handle resize functionality
  const handleResizeStart = (e: React.MouseEvent) => {
    e.preventDefault()
    resizingRef.current = true
    startXRef.current = e.clientX
    startWidthRef.current = width
    document.addEventListener("mousemove", handleResizeMove)
    document.addEventListener("mouseup", handleResizeEnd)
    document.body.style.cursor = "ew-resize"
    document.body.style.userSelect = "none"
  }

  const handleResizeMove = (e: MouseEvent) => {
    if (!resizingRef.current) return
    const deltaX = startXRef.current - e.clientX
    const newWidth = Math.min(Math.max(280, startWidthRef.current + deltaX), 600)
    setWidth(newWidth)
  }

  const handleResizeEnd = () => {
    resizingRef.current = false
    document.removeEventListener("mousemove", handleResizeMove)
    document.removeEventListener("mouseup", handleResizeEnd)
    document.body.style.cursor = ""
    document.body.style.userSelect = ""
  }

  return (
    <>
      {/* Toggle Button */}
      <Button
        size="icon"
        className="fixed top-4 right-4 z-50 rounded-full w-12 h-12 shadow-lg"
        onClick={() => setOpen((v) => !v)}
        aria-label="Toggle chat sidebar"
      >
        {open ? <X className="h-5 w-5" /> : <MessageSquare className="h-5 w-5" />}
      </Button>

      {/* Resize Handle */}
      <div
        className={cn(
          "fixed top-0 bottom-0 z-50 w-1 cursor-ew-resize transition-opacity",
          open ? "opacity-100" : "opacity-0 pointer-events-none",
        )}
        style={{ left: `calc(100% - ${width + 4}px)` }}
        onMouseDown={handleResizeStart}
      >
        <div className="absolute inset-y-0 -left-2 w-4 flex items-center justify-center opacity-0 hover:opacity-100">
          <GripVertical className="h-6 w-6 text-muted-foreground" />
        </div>
      </div>

      {/* Sidebar */}
      <div
        ref={sidebarRef}
        className={cn(
          "fixed top-0 right-0 h-full bg-background shadow-lg border-l z-40 transform transition-transform duration-300 flex flex-col",
          open ? "translate-x-0" : "translate-x-full",
        )}
        style={{ width: `${width}px` }}
      >
        <div className="flex items-center justify-between px-4 py-3 border-b bg-muted/30">
          <div className="font-medium flex items-center gap-2">
            <MessageSquare className="h-4 w-4" />
            <span>Ask about your spending</span>
          </div>
          <Button
            size="icon"
            variant="ghost"
            className="h-8 w-8"
            onClick={() => setOpen(false)}
            aria-label="Close sidebar"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 && (
            <div className="text-center text-muted-foreground py-8">
              <MessageSquare className="h-12 w-12 mx-auto mb-2 opacity-20" />
              <p>Ask a question about your spending</p>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div key={idx} className={cn("max-w-[85%] break-words", msg.role === "user" ? "ml-auto" : "mr-auto")}>
              <div
                className={cn(
                  "rounded-lg px-3 py-2",
                  msg.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted",
                )}
              >
                {msg.role === "user" ? msg.content : formatMessage(msg.content)}
              </div>
              <div
                className={cn("text-xs mt-1 text-muted-foreground", msg.role === "user" ? "text-right" : "text-left")}
              >
                {msg.role === "user" ? "You" : "Assistant"}
              </div>
            </div>
          ))}

          {loading && (
            <div className="max-w-[85%] mr-auto">
              <div className="bg-muted rounded-lg px-3 py-2 animate-pulse">Thinking...</div>
              <div className="text-xs mt-1 text-muted-foreground">Assistant</div>
            </div>
          )}

          {error && (
            <div className="max-w-[85%] mr-auto">
              <div className="bg-destructive/10 text-destructive rounded-lg px-3 py-2">{error}</div>
              <div className="text-xs mt-1 text-muted-foreground">Error</div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <form className="border-t p-3 bg-background" onSubmit={handleSubmit}>
          <div className="flex gap-2">
            <Input
              className="flex-1"
              placeholder="Ask about your expenditure..."
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              disabled={loading}
              autoComplete="off"
            />
            <Button type="submit" size="icon" disabled={loading || !userInput.trim()}>
              <Send className="h-4 w-4" />
              <span className="sr-only">Send</span>
            </Button>
          </div>
        </form>
      </div>
    </>
  )
}

export default SidebarChat