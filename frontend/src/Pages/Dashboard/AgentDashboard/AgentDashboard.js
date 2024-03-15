import React, { Component } from "react";
import eventApi from '../../../API/eventApi'
import { Route, NavLink, HashRouter } from "react-router-dom";
import TopMenuBar from "../../../Component/TopMenuBar";
import { addDays, subDays } from 'date-fns';
import Tables from "./Tables/Table"
import cookieStorage from "../../../Constants/cookie-storage";
import Access from "../../../Constants/accessValidation";
import ChangePasswordModal from "../../../Component/ChangePasswordModal";
import './AgentDashboard.scss'
import _ from 'lodash';

class AgentDashboard extends Component {
    constructor(props) {
        super(props);
        this.state = {
            displayModal: false,
            resetPass: '',
            loading: false,
            // startDate: this.formatDate(new Date()),
            // endDate: this.formatDate((new Date)),
        }
        this.regionSelected = 'Null';
        this.countrySelected = 'Null';
        this.citySelected = 'Null';
        this.gettingMonth = 'Null';
        this.gettingYear = 'Null';
        this.middleDashboardRef = React.createRef();
    }

    componentDidMount() {
        if (Access.accessValidation('Dashboard', 'decisionMakingDashboard')) {
            // this.getResetPassStatus();
            const token = cookieStorage.getCookie('Authorization')
            this.sendEvent(token);
        } else {
            this.props.history.push('/404')
        }
    }

    sendEvent = (token) => {
        var eventData = {
            event_id: "1",
            description: "User viewed Agent Dashboard Page",
            where_path: "/decisionMakingDashboard",
            page_name: "Decision Making Dashboard Dashboard Page"
        }
        eventApi.sendEventWithHeader(eventData, token)
    }

    getResetPassStatus() {
        const user = cookieStorage.getCookie('userDetails')
        const userDetails = JSON.parse(user)
        if (userDetails.is_resetpwd === 'True') {
            this.setState({ displayModal: true })
        }
    }

    getPOSSelectedGeographicalDetails = ($event) => {
        const currentMonth = window.monthNumToName(new Date().getMonth() + 1)
        this.regionSelected = $event.regionSelected
        this.countrySelected = $event.countrySelected
        this.citySelected = $event.citySelected
        this.gettingMonth = $event.gettingMonth
        this.gettingYear = $event.gettingYear

        // Call to child Component
        this.middleDashboardRef.current.setFilterValues({
            regionSelected: this.regionSelected === 'All' ? 'Null' : this.regionSelected,
            countrySelected: this.countrySelected,
            citySelected: this.citySelected,
            gettingMonth: this.gettingMonth,
            gettingYear: this.gettingYear
        });
    }

    formatDate = (d) => {
        let month = '' + (d.getMonth() + 1);
        let day = '' + d.getDate();
        let year = d.getFullYear();

        if (month.length < 2)
            month = '0' + month;
        if (day.length < 2)
            day = '0' + day;

        return [year, month, day].join('-');
    }

    render() {
        return (
            <div>
                <TopMenuBar agentDashboard={true} citySelected={this.citySelected} dashboard={true} getPOSSelectedGeographicalDetails={this.getPOSSelectedGeographicalDetails} {...this.props} />
                <Route path='/'
                    component={() => <MiddleDashboard
                        ref={this.middleDashboardRef}
                    />}
                />
                <ChangePasswordModal displayModal={this.state.displayModal} />
            </div>
        )
    }
}

class MiddleDashboard extends Component {
    constructor() {
        super();
        this.state = {
            gettingYear: 'Null',
            gettingMonth: 'Null',
            regionId: 'Null',
            cityId: 'Null',
            countryId: 'Null',
        }
    }

    setFilterValues(args) {
        const { gettingYear, gettingMonth, regionSelected, citySelected, countrySelected } = args;
        this.setState({
            gettingMonth: gettingMonth,
            regionId: regionSelected,
            cityId: citySelected,
            countryId: countrySelected,
            gettingYear: gettingYear
        })
    }

    render() {
        return (
            <div className='AgentDashboard'>
                <div className="row">
                    <Tables
                        gettingYear={this.state.gettingYear}
                        gettingMonth={this.state.gettingMonth}
                        regionId={this.state.regionId}
                        countryId={this.state.countryId}
                        cityId={this.state.cityId}
                    />
                </div>
            </div >
        );
    }
}

export default AgentDashboard;