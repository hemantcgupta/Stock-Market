import React, { Component } from "react";
import { Route, NavLink, HashRouter } from "react-router-dom";
import { addDays, subDays } from 'date-fns';
import IndicatorsPRP from "./IndicatorsPRP";
import Constant from '../../../Constants/validator';
import TopMenuBar from "../../../Component/TopMenuBar";
import Access from "../../../Constants/accessValidation";
import ChangePasswordModal from "../../../Component/ChangePasswordModal";
import eventApi from '../../../API/eventApi'
import Graphs from "./FirstSection/Graphs";
import Tables from "./SecondSection/Tables"
import cookieStorage from "../../../Constants/cookie-storage";
import _ from 'lodash';
import './PosRevenuePlanning.scss'

var date = new Date();

class PosRevenuePlanning extends Component {
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
        this.middleDashboardRef = React.createRef();
    }

    componentDidMount() {
        if (Access.accessValidation('Dashboard', 'posRevenuePlanning')) {
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
            description: "User viewed Pos Revenue Planning Dashboard Page",
            where_path: "/posRevenuePlanning",
            page_name: "Pos Revenue Planning Dashboard Page"
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
        this.startDate = $event.startDate
        this.endDate = $event.endDate

        // Call to child Component
        this.middleDashboardRef.current.setFilterValues({
            regionSelected: this.regionSelected === 'All' ? 'Null' : this.regionSelected,
            countrySelectedID: this.countrySelectedID,
            citySelected: this.citySelected,
            startDate: this.startDate,
            endDate: this.endDate
        });
    }


    render() {
        return (
            <div>
                <TopMenuBar
                    dashboard={true}
                    getPOSSelectedGeographicalDetails={this.getPOSSelectedGeographicalDetails}
                    monthRange={'nextYear'}
                    {...this.props}
                />
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
            countryId: null,
            startDate: null,
            endDate: null
        }
    }

    setFilterValues(args) {
        const { gettingMonth, regionSelected, citySelected, countrySelectedID, startDate, endDate } = args;
        this.setState({
            gettingMonth: gettingMonth,
            regionId: regionSelected,
            cityId: citySelected,
            countryId: countrySelectedID,
            startDate: startDate,
            endDate: endDate
        })
    }

    render() {
        return (
            <div className='PosRevenuePlanning'>
                <div className="row tile_count">
                    <IndicatorsPRP
                        startDate={this.state.startDate}
                        endDate={this.state.endDate}
                        regionId={this.state.regionId}
                        countryId={this.state.countryId}
                        cityId={this.state.cityId}
                        dashboard={'Pos'}
                        {...this.props}
                    />
                </div>
                <Graphs
                    url={`/rpsPos${Constant.getPOSFiltersSearchURL()}`}
                    startDate={this.state.startDate}
                    endDate={this.state.endDate}
                    regionId={this.state.regionId}
                    countryId={this.state.countryId}
                    cityId={this.state.cityId}
                    {...this.props}
                />
                <div className="row tables">
                    <Tables
                        url={`/rpsPos${Constant.getPOSFiltersSearchURL()}`}
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

export default PosRevenuePlanning;
