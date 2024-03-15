import React, { Component } from "react";
import { Route, NavLink, HashRouter } from "react-router-dom";
import { addDays, subDays } from 'date-fns';
import Indicators from "../../../Component/indicators";
import TopMenuBar from "../../../Component/TopMenuBar";
import Access from "../../../Constants/accessValidation";
import ChangePasswordModal from "../../../Component/ChangePasswordModal";
import eventApi from '../../../API/eventApi'
import Graphs from "./FirstSection/Graphs";
import Tables from "./SecondSection/Tables"
import cookieStorage from "../../../Constants/cookie-storage";
import _ from 'lodash';
import './DemographyDashboard.scss'

var date = new Date();

class DemographyDashboard extends Component {
    constructor(props) {
        super(props);
        this.state = {
            displayModal: false,
            resetPass: '',
            loading: false,
        }
        this.startDate = '';
        this.endDate = '';
        this.regionSelected = 'Null';
        this.countrySelectedID = 'Null';
        this.citySelected = 'Null';
        this.ODSelected = 'Null';
        this.middleDashboardRef = React.createRef();
    }

    componentDidMount() {
        if (Access.accessValidation('Dashboard', 'posDashboard')) {
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
            description: "User viewed demographyDashboard Dashboard Page",
            where_path: "/demographyDashboard",
            page_name: "demographyDashboard Dashboard Page"
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
        this.regionSelected = $event.regionSelected
        this.countrySelectedID = $event.countrySelected
        this.citySelected = $event.citySelected
        this.ODSelected = $event.ODSelected
        this.startDate = $event.startDate
        this.endDate = $event.endDate

        // Call to child Component
        this.middleDashboardRef.current.setFilterValues({
            regionSelected: this.regionSelected === 'All' ? 'Null' : this.regionSelected,
            countrySelectedID: this.countrySelectedID,
            citySelected: this.citySelected,
            ODSelected: this.ODSelected,
            startDate: this.startDate,
            endDate: this.endDate
        });
    }


    render() {
        return (
            <div>
                <TopMenuBar dashboard={true} Demography={true} getPOSSelectedGeographicalDetails={this.getPOSSelectedGeographicalDetails} {...this.props} />
                <MiddleDashboard
                    ref={this.middleDashboardRef}
                    {...this.props}
                />
                <ChangePasswordModal displayModal={this.state.displayModal} />
            </div>
        )
    }
}

class MiddleDashboard extends Component {
    constructor(props) {
        super(props);
        this.state = {
            gettingMonth: null,
            regionId: null,
            cityId: null,
            ODId: null,
            countryId: null,
            startDate: null,
            endDate: null
        }
    }

    setFilterValues(args) {
        const { gettingMonth, regionSelected, citySelected, ODSelected, countrySelectedID, startDate, endDate } = args;
        this.setState({
            gettingMonth: gettingMonth,
            regionId: regionSelected,
            cityId: citySelected,
            ODId: ODSelected,
            countryId: countrySelectedID,
            startDate: startDate,
            endDate: endDate
        })
    }

    render() {
        return (
            <div className='DemographyDashboard'>
                <div className="row tile_count">
                    <Indicators
                        startDate={this.state.startDate}
                        endDate={this.state.endDate}
                        regionId={this.state.regionId}
                        countryId={this.state.countryId}
                        cityId={this.state.cityId}
                        dashboard={'Demography'}
                        {...this.props}
                    />
                </div>
                <Graphs
                    startDate={this.state.startDate}
                    endDate={this.state.endDate}
                    regionId={this.state.regionId}
                    countryId={this.state.countryId}
                    cityId={this.state.cityId}
                    ODId={this.state.ODId}
                    {...this.props}
                />
                <div className="row tables">
                    <Tables
                        startDate={this.state.startDate}
                        endDate={this.state.endDate}
                        regionId={this.state.regionId}
                        countryId={this.state.countryId}
                        cityId={this.state.cityId}
                        {...this.props}
                    />
                </div>
            </div >
        );
    }
}

export default DemographyDashboard;
