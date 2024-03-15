import React from 'react';
import $ from 'jquery';
import Constants from '../../Constants/validator'
import cookieStorage from "../../Constants/cookie-storage";
import axios from 'axios';
import config from '../../Constants/config';
import loader from '../../loader.png';
import eventApi from '../../API/eventApi';


class Proceed extends React.Component {
    constructor(props) {
        super(props);

    }

    componentWillMount() {
        this.getToken();
    }

    getToken() {
        let token = Constants.getParameterByName('token', window.location.href)
        if (token) {
            cookieStorage.createCookie('Authorization', `Token ${token}`, 1)
            const headers = {
                headers: {
                    'Authorization': `Token ${token}`
                }
            }
            this.getAPIConfigData(headers)
            this.getAPIUserData(headers)
        }
    }

    getAPIConfigData(headers) {
        axios.get(`${config.API_URL}/config`, headers)
            .then((response) => {
                const menus = response.data.menus;
                cookieStorage.createCookie('menuData', JSON.stringify(menus), 1);
                cookieStorage.createCookie('events', JSON.stringify(response.data.events), 1);
            })
            .catch((error) => {
                console.log('config error', error)
            })
    }

    getAPIUserData(headers) {
        axios.get(`${config.API_URL}/loggedinuser`, headers)
            .then((response) => {
                const userData = response.data.response[0]
                cookieStorage.createCookie('userDetails', JSON.stringify(userData), 1);
                setTimeout(() => {
                    this.redirectionToDashboard()
                }, 1000);
            })
            .catch((err) => {
                console.log("user details error", err);
            })
    }

    redirectionToDashboard() {
        this.sendEvent()
        let dashboardPath = Constants.getParameterByName('whereto', window.location.href)
        this.props.history.push(`/${dashboardPath}`)
    }

    sendEvent() {
        var eventData = {
            event_id: "4",
            description: "User logged in into the system",
            where_path: "/",
            page_name: "Login Page"
        }
        eventApi.sendEvent(eventData)
    }


    render() {
        return (
            <div style={{ width: '100vw', height: '100vh', background: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <img src={loader} className="App-loader" alt="logo" />
            </div>
        );
    }
}

export default Proceed;
