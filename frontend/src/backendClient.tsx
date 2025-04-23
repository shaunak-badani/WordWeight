import axios from "axios";
import { log } from "./lib/logger"
import { globalStore } from "./store/store";

const apiURL = import.meta.env.VITE_API_ENDPOINT;
log("apiURL : ", apiURL);

const backendClient = axios.create({
    baseURL: apiURL
});

backendClient.interceptors.response.use(
    (response) => response,
    (error) => {
        if (!error.response) {
            // Backend unreachable (network error, CORS, DNS failure, etc.)
            let msg = "Cannot connect to backend server."
            globalStore.getState().setError(msg);
            return Promise.resolve({ data: null, error: { msg } });
        }
        const message =
            error.response?.data?.detail || "Something went wrong with the API.";
        
        globalStore.getState().setError(message);
        return Promise.resolve({ data: null, error: { message } });
    }
  );

export default backendClient;