import React, { Component } from 'react';
import Modal from 'react-bootstrap-modal';
import "../../.././Component/component.scss";
import eventApi from '../../.././API/eventApi'
import cookieStorage from '../../.././Constants/cookie-storage'
import APIServices from '../../.././API/apiservices'
import $ from 'jquery';
import FirstChartConfig from './FirstGraph';
import FifthChartConfig from './FifthGraph';
import SecondChartConfig from './SecondGraph';
import SixthChartConfig from './SixthGraph';
import SeventhChartConfig from './SeventhGraph';
import ThirdChartConfig from './ThirdGraph';
import EightthChartConfig from './EightthGraph';
import FourthChartConfig from './FourthGraph';


const apiServices = new APIServices();
class AllGraphs extends Component {
    constructor(props) {
        super(props);
        this.state = {
            switchforgraph: false,
            datasetOne: [],
            datasetTwo: [],
            datasetThree: [],
            datasetFourth: [],
            datasetFifth: [],
            datasetSixth: [],
            datasetSeventh: [],
            datasetEightth: [],
            loading: false,
            regionId: '*',
            countryId: '*',
            serviceGroupId: '*',
            promoTypeId: '*',
            promoTitleId: '*',
            agencyGroupId: '*',
            agentsId: '*',
            commonODId: '*',
            getCabinValueId: '*',
            type: 'Null',
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
        if (newProps.clicked) {
            this.getGraphData()
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

    toggle = () => {
        this.setState({
            switchforgraph: !this.state.switchforgraph
        }, () => this.getGraphData())
    }

    getGraphData = () => {
        const { gettingYear, gettingMonth, regionId, countryId, serviceGroupId, promoType, promoTitle, agencyGroup, agents, commonOD, getCabinValue, type } = this.props
        if (this.state.switchforgraph) {
            this.setState({
                getYear: gettingYear,
                gettingMonth: gettingMonth,
                regionId: regionId,
                countryId: countryId,
                serviceGroupId: serviceGroupId,
                promoTypeId: promoType,
                promoTitleId: promoTitle,
                agencyGroupId: agencyGroup,
                agentsId: agents,
                commonODId: commonOD,
                getCabinValue: getCabinValue,
                tabtype: type

            }
                , () => this.getPromotionSecondPageGraph())
        }
        else {
            this.setState({
                getYear: gettingYear,
                gettingMonth: gettingMonth,
                regionId: regionId,
                countryId: countryId,
                serviceGroupId: serviceGroupId,
                promoTypeId: promoType,
                promoTitleId: promoTitle,
                agencyGroupId: agencyGroup,
                agentsId: agents,
                commonODId: commonOD,
                getCabinValue: getCabinValue,
                tabtype: type
            }
                , () => this.getDDSChartData())
        }
    }

    getDDSChartData = () => {
        var self = this;
        const { getYear, currency, gettingMonth, regionId, countryId, serviceGroupId, promoTypeId, promoTitleId, agencyGroupId, agentsId, commonODId, getCabinValue, type } = this.state;

        self.showLoader();
        apiServices.getPromoTitleGraph(getYear, currency, gettingMonth, regionId, countryId, serviceGroupId, promoTypeId, promoTitleId, agencyGroupId, agentsId, commonODId, getCabinValue, type).then(function (result) {
            self.hideLoader();
            if (result) {
                var data = result[0].Data
                self.setState({ datasetOne: data })
            }
        });
        apiServices.getTicketbyChannelGraph(getYear, currency, gettingMonth, regionId, countryId, serviceGroupId, promoTypeId, promoTitleId, agencyGroupId, agentsId, commonODId, getCabinValue, type).then(function (result) {
            self.hideLoader();
            if (result) {
                var data = result[0].Data
                self.setState({ datasetTwo: data })
            }
        });
        apiServices.getTop10NonDirectionalODGraph(getYear, currency, gettingMonth, regionId, countryId, serviceGroupId, promoTypeId, promoTitleId, agencyGroupId, agentsId, commonODId, getCabinValue, type).then(function (result) {
            self.hideLoader();
            if (result) {
                var data = result[0].Data
                self.setState({ datasetThree: data })
            }
        });
        apiServices.getChannelbyRevenueGraph(getYear, currency, gettingMonth, regionId, countryId, serviceGroupId, promoTypeId, promoTitleId, agencyGroupId, agentsId, commonODId, getCabinValue, type).then(function (result) {
            self.hideLoader();
            if (result) {
                var data = result[0].Data
                self.setState({ datasetFourth: data })
            }
        });
    }

    getPromotionSecondPageGraph = () => {
        var self = this;
        const { getYear, currency, gettingMonth, regionId, countryId, serviceGroupId, promoTypeId, promoTitleId, agencyGroupId, agentsId, commonODId, getCabinValue, type } = this.state;

        self.showLoader();
        apiServices.getRevenueByIssueDateGraph(getYear, currency, gettingMonth, regionId, countryId, serviceGroupId, promoTypeId, promoTitleId, agencyGroupId, agentsId, commonODId, getCabinValue, type).then(function (result) {
            self.hideLoader();
            if (result) {
                var data = result[0].Data
                self.setState({ datasetFifth: data })
            }
        });
        apiServices.getNoOfTicketsByIssueGraph(getYear, currency, gettingMonth, regionId, countryId, serviceGroupId, promoTypeId, promoTitleId, agencyGroupId, agentsId, commonODId, getCabinValue, type).then(function (result) {
            self.hideLoader();
            if (result) {
                var data = result[0].Data
                self.setState({ datasetSeventh: data })
            }
        });
        apiServices.getRevenueByTravelPeriodGraph(getYear, currency, gettingMonth, regionId, countryId, serviceGroupId, promoTypeId, promoTitleId, agencyGroupId, agentsId, commonODId, getCabinValue, type).then(function (result) {
            self.hideLoader();
            if (result) {
                var data = result[0].Data
                self.setState({ datasetSixth: data })
            }
        });
        apiServices.getNoOfTicketsByTravelPeriodGraph(getYear, currency, gettingMonth, regionId, countryId, serviceGroupId, promoTypeId, promoTitleId, agencyGroupId, agentsId, commonODId, getCabinValue, type).then(function (result) {
            self.hideLoader();
            if (result) {
                var data = result[0].Data
                self.setState({ datasetEightth: data })
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
        const { switchforgraph } = this.state;
        return (
            <div>

                <Modal
                    show={this.props.graphVisible}
                    onHide={this.props.closeChartModal}
                    aria-labelledby="ModalHeader"
                >
                    <Modal.Header closeButton>
                        <Modal.Title id='ModalHeader'> All Graphs for Promotion Tracking </Modal.Title>
                    </Modal.Header>
                    <Modal.Body>
                        <div className='Switch_btn'>
                            <button style={{ fontSize: '20px', backgroundColor: 'orangered', color: 'white' }} onClick={() => this.toggle()}> Switch </button>
                        </div>
                        <div style={{ display: 'flex', padding: '4px' }}>
                            <div style={{ marginRight: '2px', marginLeft: '-15.5px' }}>
                                {switchforgraph == true ? <FifthChartConfig datasetFifth={this.state.datasetFifth} /> : <FirstChartConfig datasetOne={this.state.datasetOne} />}
                            </div>
                            <div>
                                {switchforgraph == true ? <SixthChartConfig datasetSixth={this.state.datasetSixth} /> : <SecondChartConfig datasetTwo={this.state.datasetTwo} />}
                            </div>
                        </div>
                        <div style={{ display: 'flex', padding: '4px' }}>
                            <div style={{ marginRight: '2px', marginLeft: '-15.5px' }}>
                                {switchforgraph == true ? <SeventhChartConfig datasetSeventh={this.state.datasetSeventh} /> : <ThirdChartConfig datasetThree={this.state.datasetThree} />}
                            </div>
                            <div>
                                {switchforgraph == true ? <EightthChartConfig datasetEightth={this.state.datasetEightth} /> : <FourthChartConfig datasetFourth={this.state.datasetFourth} />}
                            </div>
                        </div>
                    </Modal.Body>
                </Modal>

            </div>

        );
    }
}
export default AllGraphs;




