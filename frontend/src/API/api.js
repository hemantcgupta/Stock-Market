import axios from 'axios';
import $ from 'jquery';
import config from '../Constants/config';
import cookieStorage from '../Constants/cookie-storage';
import Swal from 'sweetalert2';

let api = {};
var BASE_URL = config.API_URL;

api.showLoader = () => {
    $("#loaderImage").addClass("loader-visible")
}

api.hideLoader = () => {
    $("#loaderImage").removeClass("loader-visible")
    $(".x_panel").addClass("opacity-fade");
    $(".top-buttons").addClass("opacity-fade");
}

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

api.getDefaultHeaders = getDefaultHeaders;

api.logout = () => {
    window.location = '/';
    cookieStorage.deleteCookie();
}

api.get = (url, hideLoader) => {
    let _ = hideLoader ? null : api.showLoader()
    const token = cookieStorage.getCookie('Authorization')
    return new Promise(function (resolve, reject) {
        axios.get(`${BASE_URL}/${url}`, {
            headers: {
                'Authorization': token,
                // 'bypassauth': 'yes',
                // 'user': 'vishal@rejolut.com'
            }
        })
            .then((response) => {
                api.hideLoader();
                resolve(response)
            })
            .catch((error) => {
                api.hideLoader();
                reject(error)
                api.handleError(error, reject);
            })
    });
}

api.post = (url, body, hideLoader) => {
    let _ = hideLoader ? null : api.showLoader()
    let headers = getDefaultHeaders();
    return new Promise(function (resolve, reject) {
        axios.post(`${BASE_URL}/${url}`, body, headers)
            .then((response) => {
                api.hideLoader();
                resolve(response)
            })
            .catch((error) => {
                api.hideLoader();
                reject(error)
                api.handleError(error, reject);
            })
    });
}

api.put = (url, body, hideLoader) => {
    let _ = hideLoader ? null : api.showLoader()
    let headers = getDefaultHeaders();
    return new Promise(function (resolve, reject) {
        axios.put(`${BASE_URL}/${url}`, body, headers)
            .then((response) => {
                api.hideLoader();
                resolve(response)
            }).catch((error) => {
                api.hideLoader();
                reject(error)
                api.handleError(error, reject);
            })
    });
}

api.delete = (url, data) => {
    api.showLoader();
    let headers = getDefaultHeaders();
    return new Promise(function (resolve, reject) {
        axios.delete(`${BASE_URL}/${url}`, headers)
            .then((response) => {
                api.hideLoader();
                resolve(response)
            })
            .catch((error) => {
                api.hideLoader();
                reject(error)
                api.handleError(error, reject);
            })
    });
}

api.handleError = (error, reject) => {
    if (error && error.response) {
        if (error.response.status === 403) {
            api.showAuthFail();
        }
    } else {
        Swal.fire({
            title: "Error!",
            text: "something went wrong, please try again!",
            icon: "error",
            confirmButtonText: 'Ok'
        }).then(() => {
            api.logout()
        })
    }
}

api.showAuthFail = () => {
    Swal.fire({
        title: 'Error!',
        text: 'Authorization failed!',
        icon: 'error',
        confirmButtonText: 'Ok'
    }).then(() => {
        api.logout()
    })
}

api.LoadPopup = (e) => {
    document.getElementById(e).classList.add('show')
}

api.HidePopup = (e) => {
    document.getElementById(e).classList.remove('show')
}

export default api;