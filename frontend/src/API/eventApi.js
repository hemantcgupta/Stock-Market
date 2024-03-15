import axios from 'axios';
import $ from 'jquery';
import config from '../Constants/config';
import cookieStorage from '../Constants/cookie-storage';
import Swal from 'sweetalert2';

let eventApi = {};
var BASE_URL = config.API_URL;

let getDefaultHeaders = () => {
    const token = cookieStorage.getCookie('Authorization')
    return {
        headers: {
            'Authorization': token,
            // 'bypassauth': 'yes',
            // 'user': 'vishal@rejolut.com'
        }
    }
}

eventApi.getDefaultHeaders = getDefaultHeaders;

eventApi.sendEvent = (body) => {
    let headers = getDefaultHeaders();
    return new Promise(function (resolve, reject) {
        axios.post(`${BASE_URL}/auditLogs/logs`, body, headers)
            .then((response) => {
                resolve(response)
            })
            .catch((error) => {
                eventApi.handleError(error)
                reject(error)
            })
    });
}

eventApi.sendEventWithHeader = (body, token) => {
    return new Promise(function (resolve, reject) {
        axios.post(`${BASE_URL}/auditLogs/logs`, body, {
            headers: {
                'Authorization': token,
                // 'bypassauth': 'yes',
                // 'user': 'vishal@rejolut.com'
            }
        })
            .then((response) => {
                resolve(response)
            })
            .catch((error) => {
                eventApi.handleError(error)
                reject(error)
            })
    });
}

eventApi.handleError = (error, reject) => {
    if (error && error.response) {
        if (error.response.status === 403) {
            eventApi.showAuthFail();
        }
    } else if (error.config && error.config.url) {
        Swal.fire({
            title: "Error!",
            text: "Network error, please try again!",
            icon: "error",
            confirmButtonText: 'Ok'
        }).then(() => {
            window.location.href = '/'
        })
    } else {
        // Swal.fire({
        //     title: "Error!",
        //     text: "something went wrong, please try again!",
        //     icon: "error",
        //     confirmButtonText: 'Ok'
        // }).then(() => {
        //     window.location.href = '/'
        // })
    }
}

eventApi.showAuthFail = () => {
    Swal.fire({
        title: 'Error!',
        text: 'Authorization failed! (Your token has been expired. Please login again)',
        icon: 'error',
        confirmButtonText: 'Ok'
    }).then(() => {
        window.location.href = '/'
    })
}

export default eventApi;