import React, { Component } from 'react';
import Modal from 'react-bootstrap-modal';
import "../.././Component/component.scss";
import eventApi from '../.././API/eventApi'
import cookieStorage from '../.././Constants/cookie-storage'
import APIServices from '../.././API/apiservices'
import $ from 'jquery';
import NationalityGraph from './NationalityChart';
import AgeBandGraph from './AgeBandChart';


const apiServices = new APIServices();
class AllGraphs extends Component {
    constructor(props) {
        super(props);
        this.state = {
            Nationality: false,
            datasetOne: [],
            datasetFifth: [],
            loading: false,
            regionId: '*',
            countryId: '*',
            commonOD: '*',
            cabinId: '*',
            enrichId: '*',
            getCabinValueId: '*',
            type: 'Null',
            NationalityId: '*',
            getYear: '*',
            currency: '*',
            gettingMonth: '*'
        }
    }

    sendEvent = (token) => {
        var eventData = {
            event_id: "1",
            description: "User viewed Promo Title Graph",
            where_path: "/posPromotion",
            page_name: "Pos Promotion"
        }
        eventApi.sendEventWithHeader(eventData, token)
    }

    componentDidMount() {
        const token = cookieStorage.getCookie('Authorization')
        this.sendEvent(token);
    }


    componentWillReceiveProps(newProps) {
        let { NationalityId, AgeBandId } = this.props
        if (newProps.clicked) {
            if (NationalityId !== '*' || AgeBandId !== '*') {
                this.getGraphData()
            }
        }
    }

    // getPOSSelectedGeographicalDetails = ($event) => {
    //     this.setState({
    //         regionId: $event.regionSelected,
    //         countryId: $event.countrySelected,
    //         cityId: $event.citySelected,
    //         startDate: $event.startDate,
    //         endDate: $event.endDate,
    //     }, () => this.getDDSChartData())
    // }

    // toggle = () => {
    //     this.setState({
    //         Nationality: !this.state.Nationality
    //     }, () => this.AgeBandGraph())
    // }

    getGraphData = () => {
        const { gettingYear, gettingMonth, regionId, countryId, commonOD, cabinId, enrichId, getCabinValue, type, NationalityId, AgeBandId } = this.props
        if (type === 'Nationality') {
            this.setState({
                Nationality: !this.state.Nationality,
                getYear: gettingYear,
                gettingMonth: gettingMonth,
                regionId: regionId,
                countryId: countryId,
                commonOD: commonOD,
                cabinId: cabinId,
                enrichId: enrichId,
                NationalityId: NationalityId,
                getCabinValue: getCabinValue,
                type: type

            }
                , () => this.NationalityGraph())
        }
        else if (type === 'Age Band') {
            this.setState({
                getYear: gettingYear,
                gettingMonth: gettingMonth,
                regionId: regionId,
                countryId: countryId,
                commonOD: commonOD,
                cabinId: cabinId,
                enrichId: enrichId,
                getCabinValue: getCabinValue,
                type: type,
                AgeBandId: AgeBandId
            }
                , () => this.AgeBandGraph())
        }
    }

    NationalityGraph = () => {
        var self = this;
        const { getYear, currency, gettingMonth, regionId, countryId, commonOD, cabinId, enrichId, getCabinValue, type, NationalityId } = this.state;

        self.showLoader();
        apiServices.getNationalityGraph(getYear, currency, gettingMonth, regionId, countryId, commonOD, cabinId, enrichId, getCabinValue, type, NationalityId).then(function (result) {
            self.hideLoader();
            if (result) {
                var data = result[0].Data
                self.setState({ datasetOne: data })
            }
        });
    }

    AgeBandGraph = () => {
        var self = this;
        const { getYear, currency, gettingMonth, regionId, countryId, commonOD, cabinId, enrichId, getCabinValue, type, AgeBandId } = this.state;

        self.showLoader();
        apiServices.getAgeBandGraph(getYear, currency, gettingMonth, regionId, countryId, commonOD, cabinId, enrichId, getCabinValue, type, AgeBandId).then(function (result) {
            self.hideLoader();
            if (result) {
                var data = result[0].Data
                self.setState({ datasetFifth: data })
            }
        });
    }

    showLoader = () => {
        this.setState({ loading: true })
        $("#loaderImage").addClass("loader-visible")
    }

    hideLoader = () => {
        this.setState({ loading: false })
        $("#loaderImage").removeClass("loader-visible")
        $(".x_panel").addClass("opacity-fade");
        $(".top-buttons").addClass("opacity-fade");
    }


    render() {
        const { switchforgraph, Nationality } = this.state;
        const { type } = this.props
        return (
            <div>

                <Modal
                    show={this.props.chartVisible}
                    onHide={this.props.closeChartModal}
                    aria-labelledby="ModalHeader"
                >
                    <Modal.Header closeButton>
                        {type === 'Nationality' ?
                            <Modal.Title id='ModalHeader'>  SSR Breakdown by Nationality  </Modal.Title>
                            : <Modal.Title id='ModalHeader'>  SSR Breakdown by Age Group </Modal.Title>}
                    </Modal.Header>
                    <Modal.Body>
                        {/* <div className='Switch_btn'>
                            <button style={{ fontSize: '20px', backgroundColor: 'orangered', color: 'white' }} onClick={() => this.toggle()}> Switch </button>
                        </div> */}
                        <div style={{ display: 'flex', padding: '4px' }}>
                            <div style={{ marginRight: '2px', marginLeft: '-15.5px' }}>
                                {type === 'Nationality' ? <NationalityGraph datasetOne={this.state.datasetOne} /> : <AgeBandGraph datasetFifth={this.state.datasetFifth} />}
                            </div>
                        </div>
                    </Modal.Body>
                </Modal>

            </div>

        );
    }
}
export default AllGraphs;

