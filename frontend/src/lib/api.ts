// API Client for CAD-RAG Backend

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface AnalysisRequest {
    text: string;
    enable_rag?: boolean;
    threshold?: number;
}

export interface AnalysisResponse {
    sentence: string;
    prediction: "HATEFUL" | "NOT_HATEFUL";
    confidence: number;
    category: string;
    labels: string[];
    probabilities: Record<string, number>;
    rag_context?: string;
    llm_rationale?: string;
    entities?: string[];
    neologisms?: string[];
    indicator_matches?: Record<string, string[]>;
}

export interface HealthResponse {
    status: string;
    model_loaded: boolean;
    rag_available: boolean;
}

class APIClient {
    private baseUrl: string;

    constructor(baseUrl: string = API_BASE_URL) {
        this.baseUrl = baseUrl;
    }

    async analyze(request: AnalysisRequest): Promise<AnalysisResponse> {
        const response = await fetch(`${this.baseUrl}/analyze`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(request),
        });

        if (!response.ok) {
            throw new Error(`Analysis failed: ${response.statusText}`);
        }

        return response.json();
    }

    async healthCheck(): Promise<HealthResponse> {
        const response = await fetch(`${this.baseUrl}/health`);

        if (!response.ok) {
            throw new Error(`Health check failed: ${response.statusText}`);
        }

        return response.json();
    }
}

export const api = new APIClient();
export default api;
